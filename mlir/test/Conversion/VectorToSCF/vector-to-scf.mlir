// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf))" -split-input-file -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{full-unroll=true lower-scalable=true}))" -split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefix=FULL-UNROLL
// RUN: mlir-opt %s "-convert-vector-to-scf=full-unroll target-rank=0" -split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefix=TARGET-RANK-ZERO

// CHECK-LABEL: func @vector_transfer_ops_0d(
func.func @vector_transfer_ops_0d(%M: memref<f32>) {
  %f0 = arith.constant 0.0 : f32

  // 0-d transfers are left untouched by vector-to-scf.
  // They are independently lowered to the proper memref.load/store.
  //  CHECK: vector.transfer_read {{.*}}: memref<f32>, vector<f32>
  %0 = vector.transfer_read %M[], %f0 {permutation_map = affine_map<()->()>} :
    memref<f32>, vector<f32>

  //  CHECK: vector.transfer_write {{.*}}: vector<f32>, memref<f32>
  vector.transfer_write %0, %M[] {permutation_map = affine_map<()->()>} :
    vector<f32>, memref<f32>

  return
}

// -----

// CHECK-LABEL: func @materialize_read_1d() {
func.func @materialize_read_1d() {
  %f0 = arith.constant 0.0: f32
  %A = memref.alloc () : memref<7x42xf32>
  affine.for %i0 = 0 to 7 step 4 {
    affine.for %i1 = 0 to 42 step 4 {
      %f1 = vector.transfer_read %A[%i0, %i1], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i1)
      %f2 = vector.transfer_read %A[%i0, %ip1], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip2 = affine.apply affine_map<(d0) -> (d0 + 2)> (%i1)
      %f3 = vector.transfer_read %A[%i0, %ip2], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip3 = affine.apply affine_map<(d0) -> (d0 + 3)> (%i1)
      %f4 = vector.transfer_read %A[%i0, %ip3], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      // Both accesses in the load must be clipped otherwise %i1 + 2 and %i1 + 3 will go out of bounds.
      // CHECK: scf.if
      // CHECK-NEXT: memref.load
      // CHECK-NEXT: vector.insert
      // CHECK-NEXT: scf.yield
      // CHECK-NEXT: else
      // CHECK-NEXT: scf.yield
      // Add a dummy use to prevent dead code elimination from removing transfer
      // read ops.
      "dummy_use"(%f1, %f2, %f3, %f4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL: func @materialize_read_1d_partially_specialized
func.func @materialize_read_1d_partially_specialized(%dyn1 : index, %dyn2 : index, %dyn4 : index) {
  %f0 = arith.constant 0.0: f32
  %A = memref.alloc (%dyn1, %dyn2, %dyn4) : memref<7x?x?x42x?xf32>
  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to %dyn1 {
      affine.for %i2 = 0 to %dyn2 {
        affine.for %i3 = 0 to 42 step 2 {
          affine.for %i4 = 0 to %dyn4 {
            %f1 = vector.transfer_read %A[%i0, %i1, %i2, %i3, %i4], %f0 {permutation_map = affine_map<(d0, d1, d2, d3, d4) -> (d3)>} : memref<7x?x?x42x?xf32>, vector<4xf32>
            %i3p1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i3)
            %f2 = vector.transfer_read %A[%i0, %i1, %i2, %i3p1, %i4], %f0 {permutation_map = affine_map<(d0, d1, d2, d3, d4) -> (d3)>} : memref<7x?x?x42x?xf32>, vector<4xf32>
            // Add a dummy use to prevent dead code elimination from removing
            // transfer read ops.
            "dummy_use"(%f1, %f2) : (vector<4xf32>, vector<4xf32>) -> ()
          }
        }
      }
    }
  }
  // CHECK: %[[tensor:[0-9a-zA-Z_]+]] = memref.alloc
  // CHECK-NOT: {{.*}} memref.dim %[[tensor]], %c0
  // CHECK-NOT: {{.*}} memref.dim %[[tensor]], %c3
  return
}

// -----

// CHECK: #[[$ADD:map.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func @materialize_read(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
func.func @materialize_read(%M: index, %N: index, %O: index, %P: index) {
  %f0 = arith.constant 0.0: f32
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
  // CHECK:      %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK:               %[[ALLOC:.*]] = memref.alloca() : memref<vector<5x4x3xf32>>
  // CHECK:               scf.for %[[I4:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:                 scf.if
  // CHECK:                   %[[L3:.*]] = affine.apply #[[$ADD]](%[[I3]], %[[I4]])
  // CHECK:                   scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:                     %[[VEC:.*]] = scf.for %[[I6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {{.*}} -> (vector<3xf32>) {
  // CHECK:                       %[[L0:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I6]])
  // CHECK:                       scf.if {{.*}} -> (vector<3xf32>) {
  // CHECK-NEXT:                    %[[SCAL:.*]] = memref.load %{{.*}}[%[[L0]], %[[I1]], %[[I2]], %[[L3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                    %[[RVEC:.*]] = vector.insert %[[SCAL]], %{{.*}} [%[[I6]]] : f32 into vector<3xf32>
  // CHECK-NEXT:                    scf.yield
  // CHECK-NEXT:                  } else {
  // CHECK-NEXT:                    scf.yield
  // CHECK-NEXT:                  }
  // CHECK-NEXT:                  scf.yield
  // CHECK-NEXT:                }
  // CHECK-NEXT:                memref.store %[[VEC]], {{.*}} : memref<5x4xvector<3xf32>>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            } else {
  // CHECK-NEXT:              memref.store {{.*}} : memref<5xvector<4x3xf32>>
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          %[[LD:.*]] = memref.load %[[ALLOC]][] : memref<vector<5x4x3xf32>>
  // CHECK-NEXT:          "dummy_use"(%[[LD]]) : (vector<5x4x3xf32>) -> ()
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}

  // Check that I0 + I4 (of size 3) read from first index load(L0, ...) and write into last index store(..., I4)
  // Check that I3 + I6 (of size 5) read from last index load(..., L3) and write into first index store(I6, ...)
  // Other dimensions are just accessed with I1, I2 resp.
  %A = memref.alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          %f = vector.transfer_read %A[%i0, %i1, %i2, %i3], %f0 {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, 0, d0)>} : memref<?x?x?x?xf32>, vector<5x4x3xf32>
          // Add a dummy use to prevent dead code elimination from removing
          // transfer read ops.
          "dummy_use"(%f) : (vector<5x4x3xf32>) -> ()
        }
      }
    }
  }
  return
}

// -----

// CHECK: #[[$ADD:map.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL:func @materialize_write(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
func.func @materialize_write(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-DAG:  %{{.*}} = arith.constant dense<1.000000e+00> : vector<3x4x1x5xf32>
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK:      %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT: affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:   affine.for %[[I1:.*]] = 0 to %{{.*}} step 4 {
  // CHECK-NEXT:     affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:       affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK:              %[[ALLOC:.*]] = memref.alloca() : memref<vector<3x4x1x5xf32>>
  // CHECK:              memref.store %{{.*}}, %[[ALLOC]][] : memref<vector<3x4x1x5xf32>>
  // CHECK:              %[[VECTOR_VIEW1:.*]] = vector.type_cast %[[ALLOC]] : memref<vector<3x4x1x5xf32>> to memref<3xvector<4x1x5xf32>>
  // CHECK:              scf.for %[[I4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:                scf.if
  // CHECK:                  %[[S3:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I4]])
  // CHECK:                  %[[VECTOR_VIEW2:.*]] = vector.type_cast %[[VECTOR_VIEW1]] : memref<3xvector<4x1x5xf32>> to memref<3x4xvector<1x5xf32>>
  // CHECK:                  scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:                    scf.if
  // CHECK:                      %[[S1:.*]] = affine.apply #[[$ADD]](%[[I1]], %[[I5]])
  // CHECK:                      %[[VECTOR_VIEW3:.*]] = vector.type_cast %[[VECTOR_VIEW2]] : memref<3x4xvector<1x5xf32>> to memref<3x4x1xvector<5xf32>>
  // CHECK:                      scf.for %[[I6:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
  // CHECK:                        %[[S0:.*]] = affine.apply #[[$ADD]](%[[I2]], %[[I6]])
  // CHECK:                        %[[VEC:.*]] = memref.load %[[VECTOR_VIEW3]][%[[I4]], %[[I5]], %[[I6]]] : memref<3x4x1xvector<5xf32>>
  // CHECK:                        vector.transfer_write %[[VEC]], %{{.*}}[%[[S3]], %[[S1]], %[[S0]], %[[I3]]] : vector<5xf32>, memref<?x?x?x?xf32>
  // CHECK:                      }
  // CHECK:                    }
  // CHECK:                  }
  // CHECK:                }
  // CHECK:              }
  // CHECK:            }
  // CHECK:          }
  // CHECK:        }
  // CHECK:      }
  // CHECK:      return

  // Check that I0 + I4 (of size 3) read from last index load(..., I4) and write into first index store(S0, ...)
  // Check that I1 + I5 (of size 4) read from second index load(..., I5, ...) and write into second index store(..., S1, ...)
  // Check that I3 + I6 (of size 5) read from first index load(I6, ...) and write into last index store(..., S3)
  // Other dimension is just accessed with I2.
  %A = memref.alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = arith.constant dense<1.000000e+00> : vector<5x4x3xf32>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N step 4 {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          vector.transfer_write %f1, %A[%i0, %i1, %i2, %i3] {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0)>} : vector<5x4x3xf32>, memref<?x?x?x?xf32>
        }
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>


// CHECK-LABEL: transfer_read_progressive(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index

// FULL-UNROLL-LABEL: transfer_read_progressive(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index

func.func @transfer_read_progressive(%A : memref<?x?xf32>, %base: index) -> vector<3x15xf32> {
  %f7 = arith.constant 7.0: f32
  // CHECK-DAG: %[[C7:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[splat:.*]] = arith.constant dense<7.000000e+00> : vector<15xf32>
  // CHECK-DAG: %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK:     %[[alloc_casted:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK:       %[[dim:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK:       %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:       %[[cond1:.*]] = arith.cmpi sgt, %[[dim]], %[[add]] : index
  // CHECK:       scf.if %[[cond1]] {
  // CHECK:         %[[vec_1d:.*]] = vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // CHECK:         memref.store %[[vec_1d]], %[[alloc_casted]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:       } else {
  // CHECK:         store %[[splat]], %[[alloc_casted]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:       }
  // CHECK:     }
  // CHECK:     %[[cst:.*]] = memref.load %[[alloc]][] : memref<vector<3x15xf32>>

  // FULL-UNROLL-DAG: %[[C7:.*]] = arith.constant 7.000000e+00 : f32
  // FULL-UNROLL-DAG: %[[VEC0:.*]] = arith.constant dense<7.000000e+00> : vector<3x15xf32>
  // FULL-UNROLL-DAG: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL: %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: cmpi sgt, %[[DIM]], %[[base]] : index
  // FULL-UNROLL: %[[VEC1:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%[[base]], %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC0]] [0] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: cmpi sgt, %{{.*}}, %{{.*}} : index
  // FULL-UNROLL: %[[VEC2:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC1]] [1] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: cmpi sgt, %{{.*}}, %{{.*}} : index
  // FULL-UNROLL: %[[VEC3:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC2]] [2] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }

  %f = vector.transfer_read %A[%base, %base], %f7 :
    memref<?x?xf32>, vector<3x15xf32>

  return %f: vector<3x15xf32>
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: transfer_write_progressive(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
// FULL-UNROLL-LABEL: transfer_write_progressive(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  FULL-UNROLL-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
func.func @transfer_write_progressive(%A : memref<?x?xf32>, %base: index, %vec: vector<3x15xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK:     %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK:     memref.store %[[vec]], %[[alloc]][] : memref<vector<3x15xf32>>
  // CHECK:     %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK:       %[[dim:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK:       %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:       %[[cmp:.*]] = arith.cmpi sgt, %[[dim]], %[[add]] : index
  // CHECK:       scf.if %[[cmp]] {
  // CHECK:         %[[vec_1d:.*]] = memref.load %[[vmemref]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:         vector.transfer_write %[[vec_1d]], %[[A]][{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // CHECK:       }
  // CHECK:     }

  // FULL-UNROLL: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL: %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: %[[CMP0:.*]] = arith.cmpi sgt, %[[DIM]], %[[base]] : index
  // FULL-UNROLL: scf.if %[[CMP0]] {
  // FULL-UNROLL:   %[[V0:.*]] = vector.extract %[[vec]][0] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V0]], %[[A]][%[[base]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP1:.*]] = arith.cmpi sgt, %{{.*}}, %[[I1]] : index
  // FULL-UNROLL: scf.if %[[CMP1]] {
  // FULL-UNROLL:   %[[V1:.*]] = vector.extract %[[vec]][1] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V1]], %[[A]][%{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP2:.*]] = arith.cmpi sgt, %{{.*}}, %[[I2]] : index
  // FULL-UNROLL: scf.if %[[CMP2]] {
  // FULL-UNROLL:   %[[V2:.*]] = vector.extract %[[vec]][2] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V2]], %[[A]][%{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }

  vector.transfer_write %vec, %A[%base, %base] :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: transfer_write_progressive_inbounds(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
// FULL-UNROLL-LABEL: transfer_write_progressive_inbounds(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  FULL-UNROLL-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
func.func @transfer_write_progressive_inbounds(%A : memref<?x?xf32>, %base: index, %vec: vector<3x15xf32>) {
  // CHECK-NOT:    scf.if
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK:      %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK-NEXT: memref.store %[[vec]], %[[alloc]][] : memref<vector<3x15xf32>>
  // CHECK-NEXT: %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK-NEXT:   %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK-NEXT:   %[[vec_1d:.*]] = memref.load %[[vmemref]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK-NEXT:   vector.transfer_write %[[vec_1d]], %[[A]][%[[add]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>

  // FULL-UNROLL: %[[VEC0:.*]] = vector.extract %[[vec]][0] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC0]], %[[A]][%[[base]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC1:.*]] = vector.extract %[[vec]][1] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %2, %[[A]][%[[I1]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC2:.*]] = vector.extract %[[vec]][2] : vector<15xf32> from vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC2:.*]], %[[A]][%[[I2]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  vector.transfer_write %vec, %A[%base, %base] {in_bounds = [true, true]} :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// FULL-UNROLL-LABEL: transfer_read_simple
func.func @transfer_read_simple(%A : memref<2x2xf32>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  // FULL-UNROLL-DAG: %[[VC0:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  // FULL-UNROLL-DAG: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL-DAG: %[[C1:.*]] = arith.constant 1 : index
  // FULL-UNROLL: %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]]
  // FULL-UNROLL: %[[RES0:.*]] = vector.insert %[[V0]], %[[VC0]] [0] : vector<2xf32> into vector<2x2xf32>
  // FULL-UNROLL: %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[C0]]]
  // FULL-UNROLL: %[[RES1:.*]] = vector.insert %[[V1]], %[[RES0]] [1] : vector<2xf32> into vector<2x2xf32>
  %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<2x2xf32>, vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

func.func @transfer_read_minor_identity(%A : memref<?x?x?x?xf32>) -> vector<3x3xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0, %c0, %c0], %f0
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : memref<?x?x?x?xf32>, vector<3x3xf32>
  return %0 : vector<3x3xf32>
}

// CHECK-LABEL: transfer_read_minor_identity(
//  CHECK-SAME: %[[A:.*]]: memref<?x?x?x?xf32>) -> vector<3x3xf32>
//  CHECK-DAG:    %[[c0:.*]] = arith.constant 0 : index
//  CHECK-DAG:    %[[c1:.*]] = arith.constant 1 : index
//  CHECK-DAG:    %[[c2:.*]] = arith.constant 2 : index
//  CHECK-DAG:    %[[c3:.*]] = arith.constant 3 : index
//  CHECK-DAG:    %[[f0:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG:    %[[cst0:.*]] = arith.constant dense<0.000000e+00> : vector<3xf32>
//  CHECK:        %[[m:.*]] = memref.alloca() : memref<vector<3x3xf32>>
//  CHECK:        %[[cast:.*]] = vector.type_cast %[[m]] : memref<vector<3x3xf32>> to memref<3xvector<3xf32>>
//  CHECK:        scf.for %[[arg1:.*]] = %[[c0]] to %[[c3]]
//  CHECK:          %[[d:.*]] = memref.dim %[[A]], %[[c2]] : memref<?x?x?x?xf32>
//  CHECK:          %[[cmp:.*]] = arith.cmpi sgt, %[[d]], %[[arg1]] : index
//  CHECK:          scf.if %[[cmp]] {
//  CHECK:            %[[tr:.*]] = vector.transfer_read %[[A]][%c0, %c0, %[[arg1]], %c0], %[[f0]] : memref<?x?x?x?xf32>, vector<3xf32>
//  CHECK:            memref.store %[[tr]], %[[cast]][%[[arg1]]] : memref<3xvector<3xf32>>
//  CHECK:          } else {
//  CHECK:            memref.store %[[cst0]], %[[cast]][%[[arg1]]] : memref<3xvector<3xf32>>
//  CHECK:          }
//  CHECK:        }
//  CHECK:        %[[ret:.*]]  = memref.load %[[m]][] : memref<vector<3x3xf32>>
//  CHECK:        return %[[ret]] : vector<3x3xf32>

func.func @transfer_write_minor_identity(%A : vector<3x3xf32>, %B : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  vector.transfer_write %A, %B[%c0, %c0, %c0, %c0]
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : vector<3x3xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: transfer_write_minor_identity(
// CHECK-SAME:      %[[A:.*]]: vector<3x3xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<?x?x?x?xf32>)
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[c3:.*]] = arith.constant 3 : index
// CHECK:         %[[m:.*]] = memref.alloca() : memref<vector<3x3xf32>>
// CHECK:         memref.store %[[A]], %[[m]][] : memref<vector<3x3xf32>>
// CHECK:         %[[cast:.*]] = vector.type_cast %[[m]] : memref<vector<3x3xf32>> to memref<3xvector<3xf32>>
// CHECK:         scf.for %[[arg2:.*]] = %[[c0]] to %[[c3]]
// CHECK:           %[[d:.*]] = memref.dim %[[B]], %[[c2]] : memref<?x?x?x?xf32>
// CHECK:           %[[cmp:.*]] = arith.cmpi sgt, %[[d]], %[[arg2]] : index
// CHECK:           scf.if %[[cmp]] {
// CHECK:             %[[tmp:.*]] = memref.load %[[cast]][%[[arg2]]] : memref<3xvector<3xf32>>
// CHECK:             vector.transfer_write %[[tmp]], %[[B]][%[[c0]], %[[c0]], %[[arg2]], %[[c0]]] : vector<3xf32>, memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:         }
// CHECK:         return


// -----

func.func @transfer_read_strided(%A : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0], %f0
      : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: transfer_read_strided(
// CHECK: scf.for
// CHECK: memref.load

func.func @transfer_write_strided(%A : vector<4xf32>, %B : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %A, %B[%c0, %c0] :
    vector<4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
  return
}

// CHECK-LABEL: transfer_write_strided(
// CHECK: scf.for
// CHECK: store

// -----

func.func private @fake_side_effecting_fun(%0: vector<2x2xf32>) -> ()

// CHECK-LABEL: transfer_read_within_async_execute
func.func @transfer_read_within_async_execute(%A : memref<2x2xf32>) -> !async.token {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  // CHECK-NOT: alloca
  //     CHECK: async.execute
  //     CHECK:   alloca
  %token = async.execute {
    %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<2x2xf32>, vector<2x2xf32>
    func.call @fake_side_effecting_fun(%0) : (vector<2x2xf32>) -> ()
    async.yield
  }
  return %token : !async.token
}

// -----

// CHECK-LABEL: transfer_read_with_tensor
func.func @transfer_read_with_tensor(%arg: tensor<f32>) -> vector<1xf32> {
    // CHECK:      %[[EXTRACTED:.*]] = vector.transfer_read %{{.*}}[], %{{.*}} : tensor<f32>, vector<f32>
    // CHECK-NEXT: %[[RESULT:.*]] = vector.broadcast %[[EXTRACTED]] : vector<f32> to vector<1xf32>
    // CHECK-NEXT: return %[[RESULT]] : vector<1xf32>
    %f0 = arith.constant 0.0 : f32
    %0 = vector.transfer_read %arg[], %f0 {permutation_map = affine_map<()->(0)>} :
      tensor<f32>, vector<1xf32>
    return %0: vector<1xf32>
}

// -----

// CHECK-LABEL: transfer_write_scalable
func.func @transfer_write_scalable(%arg0: memref<?xf32, strided<[?], offset: ?>>, %arg1: f32) {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xf32, strided<[?], offset: ?>>
  %1 = llvm.intr.stepvector : vector<[16]xi32>
  %2 = arith.index_cast %dim : index to i32
  %3 = llvm.mlir.undef : vector<[16]xi32>
  %4 = llvm.insertelement %2, %3[%0 : i32] : vector<[16]xi32>
  %5 = llvm.shufflevector %4, %3 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<[16]xi32>
  %6 = arith.cmpi slt, %1, %5 : vector<[16]xi32>
  %7 = llvm.mlir.undef : vector<[16]xf32>
  %8 = llvm.insertelement %arg1, %7[%0 : i32] : vector<[16]xf32>
  %9 = llvm.shufflevector %8, %7 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<[16]xf32>
  vector.transfer_write %9, %arg0[%c0], %6 {in_bounds = [true]} : vector<[16]xf32>, memref<?xf32, strided<[?], offset: ?>>
  return
}

// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-DAG:       %[[C_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[STEP:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_VEC:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : vector<[16]xi32>
// CHECK:           %[[VSCALE:.*]] = vector.vscale
// CHECK:           %[[UB:.*]] = arith.muli %[[VSCALE]], %[[C_16]] : index
// CHECK:           scf.for %[[IDX:.*]] = %[[C_0]] to %[[UB]] step %[[STEP]] {
// CHECK:             %[[MASK_VAL:.*]] = vector.extract %[[MASK_VEC]][%[[IDX]]] : i1 from vector<[16]xi1>
// CHECK:             scf.if %[[MASK_VAL]] {
// CHECK:               %[[VAL_TO_STORE:.*]] = vector.extract %{{.*}}[%[[IDX]]] : f32 from vector<[16]xf32>
// CHECK:               memref.store %[[VAL_TO_STORE]], %[[ARG_0]][%[[IDX]]] : memref<?xf32, strided<[?], offset: ?>>
// CHECK:             } else {
// CHECK:             }
// CHECK:           }

// -----

func.func @vector_print_vector_0d(%arg0: vector<f32>) {
  vector.print %arg0 : vector<f32>
  return
}
// CHECK-LABEL:   func.func @vector_print_vector_0d(
// CHECK-SAME:                                      %[[VEC:.*]]: vector<f32>) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           vector.print punctuation <open>
// CHECK:           scf.for %[[IDX:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK:             %[[EL:.*]] = vector.extract %[[VEC]][] : f32 from vector<f32>
// CHECK:             vector.print %[[EL]] : f32 punctuation <no_punctuation>
// CHECK:             %[[IS_NOT_LAST:.*]] = arith.cmpi ult, %[[IDX]], %[[C0]] : index
// CHECK:             scf.if %[[IS_NOT_LAST]] {
// CHECK:               vector.print punctuation <comma>
// CHECK:             }
// CHECK:           }
// CHECK:           vector.print punctuation <close>
// CHECK:           vector.print
// CHECK:           return
// CHECK:         }

// -----

func.func @vector_print_vector(%arg0: vector<2x2xf32>) {
  vector.print %arg0 : vector<2x2xf32>
  return
}
// CHECK-LABEL:   func.func @vector_print_vector(
// CHECK-SAME:                                   %[[VEC:.*]]: vector<2x2xf32>) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[FLAT_VEC:.*]] = vector.shape_cast %[[VEC]] : vector<2x2xf32> to vector<4xf32>
// CHECK:           vector.print punctuation <open>
// CHECK:           scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:             vector.print punctuation <open>
// CHECK:             scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:               %[[OUTER_INDEX:.*]] = arith.muli %[[I]], %[[C2]] : index
// CHECK:               %[[FLAT_INDEX:.*]] = arith.addi %[[J]], %[[OUTER_INDEX]] : index
// CHECK:               %[[EL:.*]] = vector.extract %[[FLAT_VEC]][%[[FLAT_INDEX]]] : f32 from vector<4xf32>
// CHECK:               vector.print %[[EL]] : f32 punctuation <no_punctuation>
// CHECK:               %[[IS_NOT_LAST_J:.*]] = arith.cmpi ult, %[[J]], %[[C1]] : index
// CHECK:               scf.if %[[IS_NOT_LAST_J]] {
// CHECK:                 vector.print punctuation <comma>
// CHECK:               }
// CHECK:             }
// CHECK:             vector.print punctuation <close>
// CHECK:             %[[IS_NOT_LAST_I:.*]] = arith.cmpi ult, %[[I]], %[[C1]] : index
// CHECK:             scf.if %[[IS_NOT_LAST_I]] {
// CHECK:               vector.print punctuation <comma>
// CHECK:             }
// CHECK:           }
// CHECK:           vector.print punctuation <close>
// CHECK:           vector.print
// CHECK:           return
// CHECK:         }

// -----

func.func @vector_print_scalable_vector(%arg0: vector<[4]xi32>) {
  vector.print %arg0 : vector<[4]xi32>
  return
}
// CHECK-LABEL:   func.func @vector_print_scalable_vector(
// CHECK-SAME:                                            %[[VEC:.*]]: vector<[4]xi32>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VSCALE:.*]] = vector.vscale
// CHECK:           %[[UPPER_BOUND:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK:           %[[LAST_INDEX:.*]] = arith.subi %[[UPPER_BOUND]], %[[C1]] : index
// CHECK:           vector.print punctuation <open>
// CHECK:           scf.for %[[IDX:.*]] = %[[C0]] to %[[UPPER_BOUND]] step %[[C1]] {
// CHECK:             %[[EL:.*]] = vector.extract %[[VEC]][%[[IDX]]] : i32 from vector<[4]xi32>
// CHECK:             vector.print %[[EL]] : i32 punctuation <no_punctuation>
// CHECK:             %[[IS_NOT_LAST:.*]] = arith.cmpi ult, %[[IDX]], %[[LAST_INDEX]] : index
// CHECK:             scf.if %[[IS_NOT_LAST]] {
// CHECK:               vector.print punctuation <comma>
// CHECK:             }
// CHECK:           }
// CHECK:           vector.print punctuation <close>
// CHECK:           vector.print
// CHECK:           return
// CHECK:         }

// -----

func.func @transfer_read_array_of_scalable(%arg0: memref<3x?xf32>) -> vector<3x[4]xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c1 : memref<3x?xf32>
  %mask = vector.create_mask %c1, %dim : vector<3x[4]xi1>
  %read = vector.transfer_read %arg0[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : memref<3x?xf32>, vector<3x[4]xf32>
  return %read : vector<3x[4]xf32>
}
// CHECK-LABEL:   func.func @transfer_read_array_of_scalable(
// CHECK-SAME:                                               %[[ARG:.*]]: memref<3x?xf32>) -> vector<3x[4]xf32> {
// CHECK-DAG:       %[[PADDING:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[ALLOCA_VEC:.*]] = memref.alloca() : memref<vector<3x[4]xf32>>
// CHECK:           %[[ALLOCA_MASK:.*]] = memref.alloca() : memref<vector<3x[4]xi1>>
// CHECK:           %[[DIM_SIZE:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<3x?xf32>
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[C1]], %[[DIM_SIZE]] : vector<3x[4]xi1>
// CHECK:           memref.store %[[MASK]], %[[ALLOCA_MASK]][] : memref<vector<3x[4]xi1>>
// CHECK:           %[[UNPACK_VECTOR:.*]] = vector.type_cast %[[ALLOCA_VEC]] : memref<vector<3x[4]xf32>> to memref<3xvector<[4]xf32>>
// CHECK:           %[[UNPACK_MASK:.*]] = vector.type_cast %[[ALLOCA_MASK]] : memref<vector<3x[4]xi1>> to memref<3xvector<[4]xi1>>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[MASK_SLICE:.*]] = memref.load %[[UNPACK_MASK]]{{\[}}%[[VAL_11]]] : memref<3xvector<[4]xi1>>
// CHECK:             %[[READ_SLICE:.*]] = vector.transfer_read %[[ARG]]{{\[}}%[[VAL_11]], %[[C0]]], %[[PADDING]], %[[MASK_SLICE]] {in_bounds = [true]} : memref<3x?xf32>, vector<[4]xf32>
// CHECK:             memref.store %[[READ_SLICE]], %[[UNPACK_VECTOR]]{{\[}}%[[VAL_11]]] : memref<3xvector<[4]xf32>>
// CHECK:           }
// CHECK:           %[[RESULT:.*]] = memref.load %[[ALLOCA_VEC]][] : memref<vector<3x[4]xf32>>
// CHECK:           return %[[RESULT]] : vector<3x[4]xf32>
// CHECK:         }

// -----

func.func @transfer_write_array_of_scalable(%vec: vector<3x[4]xf32>, %arg0: memref<3x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c1 : memref<3x?xf32>
  %mask = vector.create_mask %c1, %dim : vector<3x[4]xi1>
  vector.transfer_write %vec, %arg0[%c0, %c0], %mask {in_bounds = [true, true]} : vector<3x[4]xf32>, memref<3x?xf32>
  return
}
// CHECK-LABEL:   func.func @transfer_write_array_of_scalable(
// CHECK-SAME:                                                %[[VEC:.*]]: vector<3x[4]xf32>,
// CHECK-SAME:                                                %[[MEMREF:.*]]: memref<3x?xf32>) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[ALLOCA_VEC:.*]] = memref.alloca() : memref<vector<3x[4]xf32>>
// CHECK:           %[[ALLOCA_MASK:.*]] = memref.alloca() : memref<vector<3x[4]xi1>>
// CHECK:           %[[DIM_SIZE:.*]] = memref.dim %[[MEMREF]], %[[C1]] : memref<3x?xf32>
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[C1]], %[[DIM_SIZE]] : vector<3x[4]xi1>
// CHECK:           memref.store %[[MASK]], %[[ALLOCA_MASK]][] : memref<vector<3x[4]xi1>>
// CHECK:           memref.store %[[VEC]], %[[ALLOCA_VEC]][] : memref<vector<3x[4]xf32>>
// CHECK:           %[[UNPACK_VECTOR:.*]] = vector.type_cast %[[ALLOCA_VEC]] : memref<vector<3x[4]xf32>> to memref<3xvector<[4]xf32>>
// CHECK:           %[[UNPACK_MASK:.*]] = vector.type_cast %[[ALLOCA_MASK]] : memref<vector<3x[4]xi1>> to memref<3xvector<[4]xi1>>
// CHECK:           scf.for %[[VAL_11:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
// CHECK:             %[[MASK_SLICE:.*]] = memref.load %[[UNPACK_VECTOR]]{{\[}}%[[VAL_11]]] : memref<3xvector<[4]xf32>>
// CHECK:             %[[VECTOR_SLICE:.*]] = memref.load %[[UNPACK_MASK]]{{\[}}%[[VAL_11]]] : memref<3xvector<[4]xi1>>
// CHECK:             vector.transfer_write %[[MASK_SLICE]], %[[MEMREF]]{{\[}}%[[VAL_11]], %[[C0]]], %[[VECTOR_SLICE]] {in_bounds = [true]} : vector<[4]xf32>, memref<3x?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

/// The following two tests currently cannot be lowered via unpacking the leading dim since it is scalable.
/// It may be possible to special case this via a dynamic dim in future.

func.func @cannot_lower_transfer_write_with_leading_scalable(%vec: vector<[4]x4xf32>, %arg0: memref<?x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
  %mask = vector.create_mask %dim, %c4 : vector<[4]x4xi1>
  vector.transfer_write %vec, %arg0[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[4]x4xf32>, memref<?x4xf32>
  return
}
// CHECK-LABEL:   func.func @cannot_lower_transfer_write_with_leading_scalable(
// CHECK-SAME:                                                                 %[[VEC:.*]]: vector<[4]x4xf32>,
// CHECK-SAME:                                                                 %[[MEMREF:.*]]: memref<?x4xf32>)
// CHECK: vector.transfer_write %[[VEC]], %[[MEMREF]][%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]} : vector<[4]x4xf32>, memref<?x4xf32>

// -----

func.func @cannot_lower_transfer_read_with_leading_scalable(%arg0: memref<?x4xf32>) -> vector<[4]x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
  %mask = vector.create_mask %dim, %c4 : vector<[4]x4xi1>
  %read = vector.transfer_read %arg0[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : memref<?x4xf32>, vector<[4]x4xf32>
  return %read : vector<[4]x4xf32>
}
// CHECK-LABEL:   func.func @cannot_lower_transfer_read_with_leading_scalable(
// CHECK-SAME:                                                                %[[MEMREF:.*]]: memref<?x4xf32>)
// CHECK: %{{.*}} = vector.transfer_read %[[MEMREF]][%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}} {in_bounds = [true, true]} : memref<?x4xf32>, vector<[4]x4xf32>

//  -----

// Check that the `TransferOpConversion` generates valid indices for the LoadOp.

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>
func.func @does_not_crash_on_unpack_one_dim(%subview:  memref<1x1x1x1xi32>, %mask: vector<1x1xi1>) -> vector<1x1x1x1xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %3 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i32, %mask {permutation_map = #map1}
          : memref<1x1x1x1xi32>, vector<1x1x1x1xi32>
  return %3 : vector<1x1x1x1xi32>
}
// CHECK-LABEL: func.func @does_not_crash_on_unpack_one_dim
// CHECK: %[[ALLOCA_0:.*]] = memref.alloca() : memref<vector<1x1xi1>>
// CHECK: %[[MASK:.*]] = vector.type_cast %[[ALLOCA_0]] : memref<vector<1x1xi1>> to memref<1xvector<1xi1>>
// CHECK: memref.load %[[MASK]][%{{.*}}] : memref<1xvector<1xi1>>

//  -----

// Check that the `TransferOpConversion` generates valid indices for the StoreOp.
// This test is pulled from an integration test for ArmSVE.

func.func @add_arrays_of_scalable_vectors(%a: memref<1x2x?xf32>, %b: memref<1x2x?xf32>) -> vector<1x2x[4]xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 2 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim_a = memref.dim %a, %c2 : memref<1x2x?xf32>
  %mask_a = vector.create_mask %c2, %c3, %dim_a : vector<1x2x[4]xi1>
  %vector_a = vector.transfer_read %a[%c0, %c0, %c0], %cst, %mask_a {in_bounds = [true, true, true]} : memref<1x2x?xf32>, vector<1x2x[4]xf32>
  return %vector_a : vector<1x2x[4]xf32>
}
// CHECK-LABEL: func.func @add_arrays_of_scalable_vectors
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load

//  -----

// FULL-UNROLL-LABEL: @cannot_fully_unroll_transfer_write_of_nd_scalable_vector
func.func @cannot_fully_unroll_transfer_write_of_nd_scalable_vector(%vec: vector<[4]x[4]xf32>, %memref: memref<?x?xf32>) {
  // FULL-UNROLL-NOT: vector.extract
  // FULL-UNROLL: vector.transfer_write {{.*}} : vector<[4]x[4]xf32>, memref<?x?xf32>
  // FULL-UNROLL-NOT: vector.extract
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %memref[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

//  -----

// TARGET-RANK-ZERO-LABEL: func @unroll_transfer_write_target_rank_zero
func.func @unroll_transfer_write_target_rank_zero(%vec : vector<2xi32>) {
  %alloc = memref.alloc() : memref<4xi32>
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %alloc[%c0] : vector<2xi32>, memref<4xi32>
  return
}
// TARGET-RANK-ZERO: %[[ALLOC:.*]] = memref.alloc() : memref<4xi32>
// TARGET-RANK-ZERO: %[[EXTRACTED1:.*]] = vector.extract {{.*}} : i32 from vector<2xi32>
// TARGET-RANK-ZERO: %[[BROADCASTED1:.*]] = vector.broadcast %[[EXTRACTED1]] : i32 to vector<i32>
// TARGET-RANK-ZERO: vector.transfer_write %[[BROADCASTED1]], %[[ALLOC]]{{.*}} : vector<i32>, memref<4xi32>
// TARGET-RANK-ZERO: %[[EXTRACTED2:.*]] = vector.extract {{.*}} : i32 from vector<2xi32>
// TARGET-RANK-ZERO: %[[BROADCASTED2:.*]] = vector.broadcast %[[EXTRACTED2]] : i32 to vector<i32>
// TARGET-RANK-ZERO: vector.transfer_write %[[BROADCASTED2]], %[[ALLOC]]{{.*}} : vector<i32>, memref<4xi32>

// -----

func.func @scalable_transpose_store_unmasked(%vec: vector<4x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  vector.transfer_write %transpose, %dest[%i, %j] {in_bounds = [true, true]} : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL: #[[$SLICE_MAP:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// FULL-UNROLL-LABEL:   func.func @scalable_transpose_store_unmasked(
// FULL-UNROLL-SAME:                                                 %[[VEC:.*]]: vector<4x[4]xf32>,
// FULL-UNROLL-SAME:                                                 %[[DEST:.*]]: memref<?x?xf32>,
// FULL-UNROLL-SAME:                                                 %[[I:.*]]: index,
// FULL-UNROLL-SAME:                                                 %[[J:.*]]: index)
// FULL-UNROLL-DAG:       %[[C0:.*]] = arith.constant 0 : index
// FULL-UNROLL-DAG:       %[[C1:.*]] = arith.constant 1 : index
// FULL-UNROLL-DAG:       %[[C4:.*]] = arith.constant 4 : index
// FULL-UNROLL:           %[[SLICE_0:.*]] = vector.extract %[[VEC]][0] : vector<[4]xf32> from vector<4x[4]xf32>
// FULL-UNROLL:           %[[SLICE_1:.*]] = vector.extract %[[VEC]][1] : vector<[4]xf32> from vector<4x[4]xf32>
// FULL-UNROLL:           %[[SLICE_2:.*]] = vector.extract %[[VEC]][2] : vector<[4]xf32> from vector<4x[4]xf32>
// FULL-UNROLL:           %[[SLICE_3:.*]] = vector.extract %[[VEC]][3] : vector<[4]xf32> from vector<4x[4]xf32>
// FULL-UNROLL:           %[[VSCALE:.*]] = vector.vscale
// FULL-UNROLL:           %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// FULL-UNROLL:           scf.for %[[VAL_13:.*]] = %[[C0]] to %[[C4_VSCALE]] step %[[C1]] {
// FULL-UNROLL:             %[[SLICE_I:.*]] = affine.apply #[[$SLICE_MAP]](%[[VAL_13]]){{\[}}%[[I]]]
// FULL-UNROLL:             %[[ELEM_0:.*]] = vector.extract %[[SLICE_0]]{{\[}}%[[VAL_13]]] : f32 from vector<[4]xf32>
// FULL-UNROLL:             %[[ELEM_1:.*]] = vector.extract %[[SLICE_1]]{{\[}}%[[VAL_13]]] : f32 from vector<[4]xf32>
// FULL-UNROLL:             %[[ELEM_2:.*]] = vector.extract %[[SLICE_2]]{{\[}}%[[VAL_13]]] : f32 from vector<[4]xf32>
// FULL-UNROLL:             %[[ELEM_3:.*]] = vector.extract %[[SLICE_3]]{{\[}}%[[VAL_13]]] : f32 from vector<[4]xf32>
// FULL-UNROLL:             %[[TRANSPOSE_SLICE:.*]] = vector.from_elements %[[ELEM_0]], %[[ELEM_1]], %[[ELEM_2]], %[[ELEM_3]] : vector<4xf32>
// FULL-UNROLL:             vector.transfer_write %[[TRANSPOSE_SLICE]], %[[DEST]]{{\[}}%[[SLICE_I]], %[[J]]] {in_bounds = [true]} : vector<4xf32>, memref<?x?xf32>

// -----

func.func @scalable_transpose_store_dynamic_mask(%vec: vector<4x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index, %a: index, %b: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  %mask = vector.create_mask %a, %b : vector<[4]x4xi1>
  vector.transfer_write %transpose, %dest[%i, %j], %mask {in_bounds = [true, true]} : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL:   func.func @scalable_transpose_store_dynamic_mask(
// FULL-UNROLL-SAME:                                                     %{{.*}}, %[[A:.*]]: index, %[[B:.*]]: index)
// FULL-UNROLL:           %[[SLICE_MASK:.*]] = vector.create_mask %[[B]] : vector<4xi1>
// FULL-UNROLL:           scf.for %{{.*}} to %[[A]]
// FULL-UNROLL:             vector.transfer_write {{.*}}, %[[SLICE_MASK]]

// -----

func.func @scalable_transpose_store_constant_mask(%vec: vector<4x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  %mask = vector.constant_mask [4, 3] : vector<[4]x4xi1>
  vector.transfer_write %transpose, %dest[%i, %j], %mask {in_bounds = [true, true]} : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL:   func.func @scalable_transpose_store_constant_mask
// FULL-UNROLL:           %[[C3:.*]] = arith.constant 3 : index
// FULL-UNROLL:           %[[C4:.*]] = arith.constant 4 : index
// FULL-UNROLL:           %[[VSCALE:.*]] = vector.vscale
// FULL-UNROLL:           %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// FULL-UNROLL:           %[[SLICE_MASK:.*]] = vector.create_mask %[[C3]] : vector<4xi1>
// FULL-UNROLL:           scf.for %{{.*}} to %[[C4_VSCALE]]
// FULL-UNROLL:             vector.transfer_write {{.*}}, %[[SLICE_MASK]]

// -----

/// Unsupported transpose.
func.func @negative_scalable_transpose_store_0(%vec: vector<[4]x4xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<[4]x4xf32> to vector<4x[4]xf32>
  vector.transfer_write %transpose, %dest[%i, %j] {in_bounds = [true, true]} : vector<4x[4]xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL: @negative_scalable_transpose_store_0
// FULL-UNROLL-NOT:   scf.for

// -----

/// Non-identity permutation map (should be lowered first).
func.func @negative_scalable_transpose_store_1(%vec: vector<4x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  vector.transfer_write %transpose, %dest[%i, %j] {in_bounds = [true, true], permutation_map = affine_map<(d0,d1) -> (d1, d0)> } : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL: @negative_scalable_transpose_store_1
// FULL-UNROLL-NOT:   scf.for


// -----

/// Out-of-bounds dim.
func.func @negative_scalable_transpose_store_2(%vec: vector<4x[4]xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  %transpose = vector.transpose %vec, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
  vector.transfer_write %transpose, %dest[%i, %j] {in_bounds = [false, true]} : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL: @negative_scalable_transpose_store_2
// FULL-UNROLL-NOT:   scf.for

// -----

/// Source not a vector.transpose.
func.func @negative_scalable_transpose_store_3(%vec: vector<[4]x4xf32>, %dest: memref<?x?xf32>, %i: index, %j: index) {
  vector.transfer_write %vec, %dest[%i, %j] {in_bounds = [true, true]} : vector<[4]x4xf32>,  memref<?x?xf32>
  return
}
// FULL-UNROLL-LABEL: @negative_scalable_transpose_store_3
// FULL-UNROLL-NOT:   scf.for
