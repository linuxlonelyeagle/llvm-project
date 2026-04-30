// CHECK-LABEL: func @simple2(%arg0: i32, %arg1: i32, %arg2: i32) {
func.func @simple2(%arg0: i32, %arg1: i32, %arg2: i32) {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.constant 3 : i32
  return
}

