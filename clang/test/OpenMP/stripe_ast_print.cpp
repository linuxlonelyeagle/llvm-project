// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump  %s \
// RUN: | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s \
// RUN: | FileCheck %s --check-prefix=PRINT

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump-all %s \
// RUN: | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s \
// RUN: | FileCheck %s --check-prefix=PRINT

// placeholder for loop body code.
extern "C" void body(...);


// PRINT-LABEL: void foo1(
// DUMP-LABEL:  FunctionDecl {{.*}} foo1
void foo1() {
  // PRINT:     #pragma omp stripe sizes(5, 5)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  #pragma omp stripe sizes(5,5)
  // PRINT: for (int i = 7; i < 17; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 3)
    // PRINT: for (int j = 7; j < 17; j += 3)
    // DUMP:  ForStmt
    for (int j = 7; j < 17; j += 3)
    // PRINT: body(i, j);
    // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo2(
// DUMP-LABEL:  FunctionDecl {{.*}} foo2
void foo2(int start1, int start2, int end1, int end2) {
  // PRINT:     #pragma omp stripe sizes(5, 5)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  #pragma omp stripe sizes(5,5)
  // PRINT: for (int i = start1; i < end1; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = start1; i < end1; i += 1)
    // PRINT: for (int j = start2; j < end2; j += 1)
    // DUMP:  ForStmt
    for (int j = start2; j < end2; j += 1)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo3(
// DUMP-LABEL:  FunctionDecl {{.*}} foo3
void foo3() {
  // PRINT: #pragma omp for
  // DUMP:  OMPForDirective
  // DUMP-NEXT:    CapturedStmt
  // DUMP-NEXT:      CapturedDecl
  #pragma omp for
  // PRINT:     #pragma omp stripe sizes(5)
  // DUMP-NEXT:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  #pragma omp stripe sizes(5)
  for (int i = 7; i < 17; i += 3)
    // PRINT: body(i);
    // DUMP:  CallExpr
    body(i);
}


// PRINT-LABEL: void foo4(
// DUMP-LABEL:  FunctionDecl {{.*}} foo4
void foo4() {
  // PRINT: #pragma omp for collapse(3)
  // DUMP: OMPForDirective
  // DUMP-NEXT: OMPCollapseClause
  // DUMP-NEXT:  ConstantExpr
  // DUMP-NEXT:  value: Int 3
  // DUMP-NEXT:  IntegerLiteral {{.*}} 3
  // DUMP-NEXT:    CapturedStmt
  // DUMP-NEXT:      CapturedDecl
  #pragma omp for collapse(3)
  // PRINT:     #pragma omp stripe sizes(5, 5)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  // DUMP-NEXT:     IntegerLiteral {{.*}} 5
  #pragma omp stripe sizes(5, 5)
  // PRINT: for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT: for (int j = 7; j < 17; j += 1)
    // DUMP:  ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo5(
// DUMP-LABEL:  FunctionDecl {{.*}} foo5
void foo5(int start, int end, int step) {
  // PRINT: #pragma omp for collapse(2)
  // DUMP:      OMPForDirective
  // DUMP-NEXT:   OMPCollapseClause
  // DUMP-NEXT:    ConstantExpr
  // DUMP-NEXT:      value: Int 2
  // DUMP-NEXT:    IntegerLiteral {{.*}} 2
  // DUMP-NEXT:  CapturedStmt
  // DUMP-NEXT:    CapturedDecl
  #pragma omp for collapse(2)
  // PRINT: for (int i = 7; i < 17; i += 1)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 17; i += 1)
    // PRINT:     #pragma omp stripe sizes(5)
    // DUMP:      OMPStripeDirective
    // DUMP-NEXT:   OMPSizesClause
    // DUMP-NEXT:     IntegerLiteral {{.*}} 5
    #pragma omp stripe sizes(5)
    // PRINT: for (int j = 7; j < 17; j += 1)
    // DUMP-NEXT: ForStmt
    for (int j = 7; j < 17; j += 1)
      // PRINT: body(i, j);
      // DUMP:  CallExpr
      body(i, j);
}


// PRINT-LABEL: void foo6(
// DUMP-LABEL: FunctionTemplateDecl {{.*}} foo6
template<typename T, T Step, T Stripe>
void foo6(T start, T end) {
  // PRINT: #pragma omp stripe sizes(Stripe)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     DeclRefExpr {{.*}} 'Stripe' 'T'
  #pragma omp stripe sizes(Stripe)
    // PRINT-NEXT:  for (T i = start; i < end; i += Step)
    // DUMP-NEXT: ForStmt
    for (T i = start; i < end; i += Step)
      // PRINT-NEXT: body(i);
      // DUMP:  CallExpr
      body(i);
}

// Also test instantiating the template.
void tfoo6() {
  foo6<int,3,5>(0, 42);
}


// PRINT-LABEL: template <int Stripe> void foo7(int start, int stop, int step) {
// DUMP-LABEL: FunctionTemplateDecl {{.*}} foo7
template <int Stripe>
void foo7(int start, int stop, int step) {
  // PRINT: #pragma omp stripe sizes(Stripe)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     DeclRefExpr {{.*}} 'Stripe' 'int'
  #pragma omp stripe sizes(Stripe)
    // PRINT-NEXT:  for (int i = start; i < stop; i += step)
    // DUMP-NEXT: ForStmt
    for (int i = start; i < stop; i += step)
      // PRINT-NEXT: body(i);
      // DUMP:  CallExpr
      body(i);
}
void tfoo7() {
  foo7<5>(0, 42, 2);
}


// PRINT-LABEL: void foo8(
// DUMP-LABEL:  FunctionDecl {{.*}} foo8
void foo8(int a) {
  // PRINT:     #pragma omp stripe sizes(a)
  // DUMP:      OMPStripeDirective
  // DUMP-NEXT:   OMPSizesClause
  // DUMP-NEXT:     ImplicitCastExpr
  // DUMP-NEXT:       DeclRefExpr {{.*}} 'a'
  #pragma omp stripe sizes(a)
  // PRINT-NEXT: for (int i = 7; i < 19; i += 3)
  // DUMP-NEXT: ForStmt
  for (int i = 7; i < 19; i += 3)
    // PRINT: body(i);
    // DUMP:  CallExpr
    body(i);
}
