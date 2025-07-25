; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py UTC_ARGS: --version 5
; RUN: opt -passes="print<access-info>" -disable-output %s 2>&1 | FileCheck %s

define i32 @completely_before_or_after_true_dep_different_size(ptr %d) {
; CHECK-LABEL: 'completely_before_or_after_true_dep_different_size'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.128 = getelementptr i8, ptr %d, i64 128
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.128.iv = getelementptr i64, ptr %gep.128, i64 %iv
  store i64 0, ptr %gep.128.iv, align 8
  %gep.iv = getelementptr i32, ptr %d, i64 %iv
  %l = load i32, ptr %gep.iv, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 32
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %l
}

define i32 @may_overlap_true_dep_different_size(ptr %d) {
; CHECK-LABEL: 'may_overlap_true_dep_different_size'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
; CHECK-NEXT:        Comparing group GRP0:
; CHECK-NEXT:          %gep.128.iv = getelementptr i64, ptr %gep.128, i64 %iv
; CHECK-NEXT:        Against group GRP1:
; CHECK-NEXT:          %gep.iv = getelementptr i32, ptr %d, i64 %iv
; CHECK-NEXT:      Grouped accesses:
; CHECK-NEXT:        Group GRP0:
; CHECK-NEXT:          (Low: (127 + %d) High: (383 + %d))
; CHECK-NEXT:            Member: {(127 + %d),+,8}<nw><%loop>
; CHECK-NEXT:        Group GRP1:
; CHECK-NEXT:          (Low: %d High: (128 + %d))
; CHECK-NEXT:            Member: {%d,+,4}<nw><%loop>
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.128 = getelementptr i8, ptr %d, i64 127
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.128.iv = getelementptr i64, ptr %gep.128, i64 %iv
  store i64 0, ptr %gep.128.iv, align 8
  %gep.iv= getelementptr i32, ptr %d, i64 %iv
  %l = load i32, ptr %gep.iv, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 32
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %l
}

define void @completely_after_stores_with_different_sizes(ptr %dst) {
; CHECK-LABEL: 'completely_after_stores_with_different_sizes'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.dst.128 = getelementptr nuw i8, ptr %dst, i64 128
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.iv = getelementptr i16, ptr %dst, i64 %iv
  store i16 0, ptr %gep.iv, align 2
  %gep.dst.128.iv = getelementptr i8, ptr %gep.dst.128, i64 %iv
  store i8 0, ptr %gep.dst.128.iv, align 1
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 64
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}


define void @may_overlap_stores_with_different_sizes(ptr %dst) {
; CHECK-LABEL: 'may_overlap_stores_with_different_sizes'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe with run-time checks
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Check 0:
; CHECK-NEXT:        Comparing group GRP0:
; CHECK-NEXT:          %gep.iv = getelementptr i16, ptr %dst, i64 %iv
; CHECK-NEXT:        Against group GRP1:
; CHECK-NEXT:          %gep.dst.128.iv = getelementptr i8, ptr %gep.dst.128, i64 %iv
; CHECK-NEXT:      Grouped accesses:
; CHECK-NEXT:        Group GRP0:
; CHECK-NEXT:          (Low: %dst High: (130 + %dst))
; CHECK-NEXT:            Member: {%dst,+,2}<nw><%loop>
; CHECK-NEXT:        Group GRP1:
; CHECK-NEXT:          (Low: (128 + %dst)<nuw> High: (193 + %dst))
; CHECK-NEXT:            Member: {(128 + %dst)<nuw>,+,1}<nw><%loop>
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.dst.128 = getelementptr nuw i8, ptr %dst, i64 128
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.iv = getelementptr i16, ptr %dst, i64 %iv
  store i16 0, ptr %gep.iv, align 2
  %gep.dst.128.iv = getelementptr i8, ptr %gep.dst.128, i64 %iv
  store i8 0, ptr %gep.dst.128.iv, align 1
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 65
  br i1 %ec, label %exit, label %loop

exit:                            ; preds = %loop
  ret void
}

define void @completely_before_or_after_non_const_distance(ptr %dst) {
; CHECK-LABEL: 'completely_before_or_after_non_const_distance'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.off = getelementptr i8, ptr %dst, i64 576
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.mul = mul i64 %iv, 84
  %gep.404 = getelementptr i8, ptr %dst, i64 404
  %gep.iv.mul = getelementptr i8, ptr %gep.404, i64 %iv.mul
  store i32 0, ptr %gep.iv.mul, align 4
  %gep.off.iv = getelementptr i32, ptr %gep.off, i64 %iv
  store i32 0, ptr %gep.off.iv, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 3
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @overlap_non_const_distance(ptr %dst) {
; CHECK-LABEL: 'overlap_non_const_distance'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Report: unsafe dependent memory operations in loop. Use #pragma clang loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT:  Unknown data dependence.
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        Unknown:
; CHECK-NEXT:            store i32 0, ptr %gep.iv.mul, align 4 ->
; CHECK-NEXT:            store i32 0, ptr %gep.off.iv, align 4
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  %gep.off = getelementptr i8, ptr %dst, i64 575
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.mul = mul i64 %iv, 84
  %gep.404 = getelementptr i8, ptr %dst, i64 404
  %gep.iv.mul = getelementptr i8, ptr %gep.404, i64 %iv.mul
  store i32 0, ptr %gep.iv.mul, align 4
  %gep.off.iv = getelementptr i32, ptr %gep.off, i64 %iv
  store i32 0, ptr %gep.off.iv, align 4
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 3
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @accesses_completely_before_or_after_instead_backwards_vectorizable(ptr dereferenceable(800) %dst) {
; CHECK-LABEL: 'accesses_completely_before_or_after_instead_backwards_vectorizable'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %mul.2 = shl i64 %iv, 1
  %gep.mul.2 = getelementptr i16, ptr %dst, i64 %mul.2
  store i16 0, ptr %gep.mul.2, align 2
  %iv.32 = add i64 %iv, 32
  %gep.iv.32 = getelementptr i16, ptr %dst, i64 %iv.32
  store i16 0, ptr %gep.iv.32, align 2
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 15
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @accesses_may_overlap_backwards_vectorizable(ptr dereferenceable(800) %dst) {
; CHECK-LABEL: 'accesses_may_overlap_backwards_vectorizable'
; CHECK-NEXT:    loop:
; CHECK-NEXT:      Memory dependences are safe with a maximum safe vector width of 128 bits
; CHECK-NEXT:      Dependences:
; CHECK-NEXT:        BackwardVectorizable:
; CHECK-NEXT:            store i16 0, ptr %gep.mul.2, align 2 ->
; CHECK-NEXT:            store i16 0, ptr %gep.iv.32, align 2
; CHECK-EMPTY:
; CHECK-NEXT:      Run-time memory checks:
; CHECK-NEXT:      Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:      SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:      Expressions re-written:
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %mul.2 = shl i64 %iv, 1
  %gep.mul.2 = getelementptr i16, ptr %dst, i64 %mul.2
  store i16 0, ptr %gep.mul.2, align 2
  %iv.32 = add i64 %iv, 32
  %gep.iv.32 = getelementptr i16, ptr %dst, i64 %iv.32
  store i16 0, ptr %gep.iv.32, align 2
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 16
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
