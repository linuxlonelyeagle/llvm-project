// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:PointerEscape=true -analyzer-config debug.AnalysisOrder:PostCall=true %s 2>&1 | FileCheck %s


void f(int *);
int *getMem(void);

int main(void) {
    f(getMem());
    return 0;
}

// CHECK: PostCall (f)
// CHECK-NEXT: PointerEscape
