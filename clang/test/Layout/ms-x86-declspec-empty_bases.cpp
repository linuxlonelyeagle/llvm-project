// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -triple amdgcn-amd-amdhsa -fms-extensions \
// RUN:    -target-cpu gfx1200 -aux-triple x86_64-pc-win32 -fcuda-is-device \
// RUN:    -fdump-record-layouts -fsyntax-only -xhip %s 2>/dev/null \
// RUN:    | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -triple spirv64-amd-amdhsa -fms-extensions \
// RUN:    -aux-triple x86_64-pc-win32 -fcuda-is-device \
// RUN:    -fdump-record-layouts -fsyntax-only -xhip %s 2>/dev/null \
// RUN:    | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -triple nvptx64 -fms-extensions \
// RUN:    -aux-triple x86_64-pc-win32 -fcuda-is-device -fdump-record-layouts \
// RUN:    -fsyntax-only -xcuda %s 2>/dev/null | FileCheck %s

namespace test1 {

struct A {
  int a;
};
struct B {
  int b;
};
struct C {};
struct __declspec(align(16)) D {};
struct __declspec(empty_bases) X : A, D, B, C {
};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::D (empty)
// CHECK-NEXT:            | [sizeof=16, align=16,
// CHECK-NEXT:            |  nvsize=0, nvalign=16]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::B
// CHECK-NEXT:          0 |   int b
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::C (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::X
// CHECK-NEXT:          0 |   struct test1::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          0 |   struct test1::D (base) (empty)
// CHECK-NEXT:          0 |   struct test1::C (base) (empty)
// CHECK-NEXT:          4 |   struct test1::B (base)
// CHECK-NEXT:          4 |     int b
// CHECK-NEXT:            | [sizeof=16, align=16,
// CHECK-NEXT:            |  nvsize=16, nvalign=16]

int _ = sizeof(X);
}

namespace test2 {
struct A {
  int a;
};
struct __declspec(empty_bases) B {};
struct C : A {
  B b;
};

struct D {};
struct E {
  int e;
};
struct F : D, E {};

struct G : C, F {};

int _ = sizeof(G);

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::B (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::C
// CHECK-NEXT:          0 |   struct test2::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          4 |   struct test2::B b (empty)
// CHECK-NEXT:            | [sizeof=8, align=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::D (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::E
// CHECK-NEXT:          0 |   int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::F
// CHECK-NEXT:          0 |   struct test2::D (base) (empty)
// CHECK-NEXT:          0 |   struct test2::E (base)
// CHECK-NEXT:          0 |     int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::G
// CHECK-NEXT:          0 |   struct test2::C (base)
// CHECK-NEXT:          0 |     struct test2::A (base)
// CHECK-NEXT:          0 |       int a
// CHECK-NEXT:          4 |     struct test2::B b (empty)
// CHECK-NEXT:          8 |   struct test2::F (base)
// CHECK-NEXT:          8 |     struct test2::D (base) (empty)
// CHECK-NEXT:          8 |     struct test2::E (base)
// CHECK-NEXT:          8 |       int e
// CHECK-NEXT:            | [sizeof=12, align=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4]
}

namespace test3 {
struct A {
  int a;
};
struct B {};
struct C : A {
  B b;
};

struct D {};
struct E {
  int e;
};
struct F : D, E {};

struct __declspec(empty_bases) G : C, F {};

int _ = sizeof(G);

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::B (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::C
// CHECK-NEXT:          0 |   struct test3::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          4 |   struct test3::B b (empty)
// CHECK-NEXT:            | [sizeof=8, align=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::D (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::E
// CHECK-NEXT:          0 |   int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::F
// CHECK-NEXT:          0 |   struct test3::D (base) (empty)
// CHECK-NEXT:          0 |   struct test3::E (base)
// CHECK-NEXT:          0 |     int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::G
// CHECK-NEXT:          0 |   struct test3::C (base)
// CHECK-NEXT:          0 |     struct test3::A (base)
// CHECK-NEXT:          0 |       int a
// CHECK-NEXT:          4 |     struct test3::B b (empty)
// CHECK-NEXT:          8 |   struct test3::F (base)
// CHECK-NEXT:          8 |     struct test3::D (base) (empty)
// CHECK-NEXT:          8 |     struct test3::E (base)
// CHECK-NEXT:          8 |       int e
// CHECK-NEXT:            | [sizeof=12, align=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4]
}

namespace test4 {
struct A {
  int a;
};
struct B {};
struct C : A {
  B b;
};

struct __declspec(empty_bases) D {};
struct E {
  int e;
};
struct F : D, E {};

struct G : C, F {};

int _ = sizeof(G);

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::B (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::C
// CHECK-NEXT:          0 |   struct test4::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          4 |   struct test4::B b (empty)
// CHECK-NEXT:            | [sizeof=8, align=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::D (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::E
// CHECK-NEXT:          0 |   int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::F
// CHECK-NEXT:          0 |   struct test4::D (base) (empty)
// CHECK-NEXT:          0 |   struct test4::E (base)
// CHECK-NEXT:          0 |     int e
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::G
// CHECK-NEXT:          0 |   struct test4::C (base)
// CHECK-NEXT:          0 |     struct test4::A (base)
// CHECK-NEXT:          0 |       int a
// CHECK-NEXT:          4 |     struct test4::B b (empty)
// CHECK-NEXT:          8 |   struct test4::F (base)
// CHECK-NEXT:          8 |     struct test4::D (base) (empty)
// CHECK-NEXT:          8 |     struct test4::E (base)
// CHECK-NEXT:          8 |       int e
// CHECK-NEXT:            | [sizeof=12, align=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4]
}

namespace test5 {

struct A {
  int a;
};
struct B {
  int b;
};
struct C {};
struct __declspec(align(16)) D {};
struct E {
  [[msvc::no_unique_address]] C c;
};
struct __declspec(empty_bases) X : A, D, B, C, E {
};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::D (empty)
// CHECK-NEXT:            | [sizeof=16, align=16,
// CHECK-NEXT:            |  nvsize=0, nvalign=16]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::B
// CHECK-NEXT:          0 |   int b
// CHECK-NEXT:            | [sizeof=4, align=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::C (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=0, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::E (empty)
// CHECK-NEXT:          0 |   struct test5::C c (empty)
// CHECK-NEXT:            | [sizeof=1, align=1,
// CHECK-NEXT:            |  nvsize=1, nvalign=1]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::X
// CHECK-NEXT:          0 |   struct test5::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          0 |   struct test5::D (base) (empty)
// CHECK-NEXT:          0 |   struct test5::C (base) (empty)
// CHECK-NEXT:          4 |   struct test5::B (base)
// CHECK-NEXT:          4 |     int b
// CHECK-NEXT:          8 |   struct test5::E (base) (empty)
// CHECK-NEXT:          8 |     struct test5::C c (empty)
// CHECK-NEXT:            | [sizeof=16, align=16,
// CHECK-NEXT:            |  nvsize=16, nvalign=16]

int _ = sizeof(X);
}
