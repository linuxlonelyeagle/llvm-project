//===- LoopCanonicalization.cpp - Cross-dialect canonicalization patterns -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains cross-dialect canonicalization patterns that cannot be
// actual canonicalization patterns due to undesired additional dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORLOOPCANONICALIZATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// A simple, conservative analysis to determine if the loop is shape
/// conserving. I.e., the type of the arg-th yielded value is the same as the
/// type of the corresponding basic block argument of the loop.
/// Note: This function handles only simple cases. Expand as needed.
static bool isShapePreserving(ForOp forOp, int64_t arg) {
  assert(arg < static_cast<int64_t>(forOp.getNumResults()) &&
         "arg is out of bounds");
  Value value = forOp.getYieldedValues()[arg];
  while (value) {
    if (value == forOp.getRegionIterArgs()[arg])
      return true;
    OpResult opResult = dyn_cast<OpResult>(value);
    if (!opResult)
      return false;

    using tensor::InsertSliceOp;
    value = llvm::TypeSwitch<Operation *, Value>(opResult.getOwner())
                .template Case<InsertSliceOp>(
                    [&](InsertSliceOp op) { return op.getDest(); })
                .template Case<ForOp>([&](ForOp forOp) {
                  return isShapePreserving(forOp, opResult.getResultNumber())
                             ? forOp.getInitArgs()[opResult.getResultNumber()]
                             : Value();
                })
                .Default(nullptr);
  }
  return false;
}

namespace {
/// Fold dim ops of iter_args to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
///   ...
/// }
/// ```
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
template <typename OpTy>
struct DimOfIterArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!blockArg)
      return failure();
    auto forOp = dyn_cast<ForOp>(blockArg.getParentBlock()->getParentOp());
    if (!forOp)
      return failure();
    if (!isShapePreserving(forOp, blockArg.getArgNumber() - 1))
      return failure();

    Value initArg = forOp.getTiedLoopInit(blockArg)->get();
    rewriter.modifyOpInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(initArg); });

    return success();
  };
};

/// Fold dim ops of loop results to dim ops of their respective init args. E.g.:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %r, %c0 : tensor<?x?xf32>
/// ```
///
/// is folded to:
///
/// ```
/// %0 = ... : tensor<?x?xf32>
/// %r = scf.for ... iter_args(%arg0 = %0) -> (tensor<?x?xf32>) {
///   ...
/// }
/// %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
/// ```
///
/// Note: Dim ops are folded only if it can be proven that the runtime type of
/// the iter arg does not change with loop iterations.
template <typename OpTy>
struct DimOfLoopResultFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy dimOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dimOp.getSource().template getDefiningOp<scf::ForOp>();
    if (!forOp)
      return failure();
    auto opResult = cast<OpResult>(dimOp.getSource());
    unsigned resultNumber = opResult.getResultNumber();
    if (!isShapePreserving(forOp, resultNumber))
      return failure();
    rewriter.modifyOpInPlace(dimOp, [&]() {
      dimOp.getSourceMutable().assign(forOp.getInitArgs()[resultNumber]);
    });
    return success();
  }
};

/// Canonicalize AffineMinOp/AffineMaxOp operations in the context of scf.for
/// and scf.parallel loops with a known range.
template <typename OpTy>
struct AffineOpSCFCanonicalizationPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return scf::canonicalizeMinMaxOpInLoop(rewriter, op, scf::matchForLikeLoop);
  }
};

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(PatternRewriter &rewriter, scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;
  ValueBoundsConstraintSet::Variable lower(rewriter.getSymbolIdentityMap(),
                                           op.getLowerBound());
  ValueBoundsConstraintSet::Variable upper(rewriter.getSymbolIdentityMap(),
                                           op.getUpperBound());
  FailureOr<bool> isLb = ValueBoundsConstraintSet::compare(
      lower, ValueBoundsConstraintSet::LT, upper);
  return isLb.value_or(false);
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(PatternRewriter &rewriter, scf::ForOp op) {
  // Can't perform the analysis if the loops's bounds aren't index-typed.
  if (!op.getInductionVar().getType().isIndex())
    return false;

  // The loop will only loop once if the inducation variable for the next time
  // in the loop is greater than or equal to upper.
  MLIRContext *context = op.getContext();
  SmallVector<Value> nextIterOperands = {op.getLowerBound(), op.getStep()};
  AffineExpr nextIterExpr =
      rewriter.getAffineSymbolExpr(0) + rewriter.getAffineSymbolExpr(1);
  AffineMap nextItMap = AffineMap::get(0, 2, nextIterExpr);
  ValueBoundsConstraintSet::Variable nextItVar(nextItMap, nextIterOperands);
  ValueBoundsConstraintSet::Variable upperVar(rewriter.getSymbolIdentityMap(),
                                              op.getUpperBound());
  FailureOr<bool> isUpperUnderNextIter = ValueBoundsConstraintSet::compare(
      nextItVar, ValueBoundsConstraintSet::LE, upperVar);
  return isUpperUnderNextIter.value_or(false);
}

struct SCFForLoopValueBoundCanonicalizationPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!(alwaysRunsFirstIteration(rewriter, op) &&
          neverRunsSecondIteration(rewriter, op))) {
      return failure();
    }

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value> blockArgs;
    blockArgs.reserve(op.getInits().size() + 1);
    rewriter.setInsertionPointToStart(op.getBody());
    Value lower = op.getLowerBound();
    op.getInductionVar().replaceAllUsesWith(lower);
    blockArgs.push_back(lower);
    llvm::append_range(blockArgs, op.getInits());
    replaceOpWithRegion(rewriter, op, op.getRegion(), blockArgs);
    return success();
  }
};

struct SCFForLoopCanonicalization
    : public impl::SCFForLoopCanonicalizationBase<SCFForLoopCanonicalization> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(parentOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::scf::populateSCFForLoopCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns
      .add<AffineOpSCFCanonicalizationPattern<affine::AffineMinOp>,
           AffineOpSCFCanonicalizationPattern<affine::AffineMaxOp>,
           DimOfIterArgFolder<tensor::DimOp>, DimOfIterArgFolder<memref::DimOp>,
           DimOfLoopResultFolder<tensor::DimOp>,
           DimOfLoopResultFolder<memref::DimOp>,
           SCFForLoopValueBoundCanonicalizationPattern>(ctx);
}

std::unique_ptr<Pass> mlir::createSCFForLoopCanonicalizationPass() {
  return std::make_unique<SCFForLoopCanonicalization>();
}
