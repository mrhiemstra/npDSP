import numpy as np

from npdsp import *


def test_absolute() -> None:
    block = Absolute()

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 1
    np.testing.assert_array_equal(block.process(np.asarray([1, -1])),
                                  [1,1])

    assert block.process(np.asarray(-1)) == 1
    assert block.process(np.asarray(np.complex64(1,1))).astype(np.float32) == np.sqrt(2).astype(np.float32)
    assert block.process(np.asarray(np.complex64(0,1))) == 1
    assert block.process(np.asarray(np.complex64(-1,-1))).astype(np.float32) == np.sqrt(2).astype(np.float32)

def test_add() -> None:
    block_real = Add(value=5)

    assert block_real.process(np.asarray(0)) == 5
    assert block_real.process(np.asarray(1)) == 6
    assert block_real.process(np.asarray(-1)) == 4

    block_complex = Add(value=np.complex64(5,5))

    assert block_complex.process(np.asarray(np.complex64(1,1))) == np.complex64(6,6)
    assert block_complex.process(np.asarray(np.complex64(0,1))) == np.complex64(5,6)
    assert block_complex.process(np.asarray(np.complex64(-1,-1))) == np.complex64(4,4)

    block_multiple_values= Add(value=[1,2,3])
    np.testing.assert_array_equal(block_multiple_values.process(np.asarray([1,2,3])),
                                  [2,4,6])
    np.testing.assert_array_equal(block_multiple_values.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[2,3,4],[3,4,5],[4,5,6]])

    block_multiple_values2= Add(value=[[1],[2],[3]])
    np.testing.assert_array_equal(block_multiple_values2.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[2,2,2],[4,4,4],[6,6,6]])


def test_clip() -> None:
    block_real = Clip(bounds=(-1,1))

    assert block_real.process(np.asarray(-2)) == -1
    assert block_real.process(np.asarray(-1)) == -1
    assert block_real.process(np.asarray(0)) == 0
    assert block_real.process(np.asarray(1)) == 1
    assert block_real.process(np.asarray(2)) == 1

def test_conjugate() -> None:
    block = Conjugate()

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 1
    assert block.process(np.asarray(-1)) == -1
    assert block.process(np.asarray(np.complex64(1,1))) == np.complex64(1,-1)
    assert block.process(np.asarray(np.complex64(0,1))) == np.complex64(0,-1)
    assert block.process(np.asarray(np.complex64(-1,-1))) == np.complex64(-1,1)
    np.testing.assert_array_equal(block.process(np.asarray([np.complex64(1,1),np.complex64(1,-1),np.complex64(1,0)])),
                                  [np.complex64(1,-1),np.complex64(1,1),np.complex64(1,0)])

def test_divide() -> None:
    block = Divide(value=5)

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 0.2
    assert block.process(np.asarray(-1)) == -0.2
    assert block.process(np.asarray(np.complex64(1,1))) == np.complex64(0.2,0.2)
    assert block.process(np.asarray(np.complex64(0,1))) == np.complex64(0,0.2)
    assert block.process(np.asarray(np.complex64(-1,-1))) == np.complex64(-0.2,-0.2)

    np.testing.assert_array_equal(block.process(np.asarray([1,2,3])),
                                  [0.2,0.4,0.6])
    np.testing.assert_array_equal(block.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[0.2,0.2,0.2],[0.4,0.4,0.4],[0.6,0.6,0.6]])

def test_floor() -> None:
    block = Floor(value=5)

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 0
    assert block.process(np.asarray(-1)) == -1
    assert block.process(np.asarray(5)) == 1
    np.testing.assert_array_equal(block.process(np.asarray([1,2,4,6])),
                                  [0,0,0,1])

def test_maximum() -> None:
    block = Maximum()
    assert block.process(np.asarray([0,1,2,3])) == 3
    assert block.process(np.asarray([np.complex64(1,1),np.complex64(2,2)])) == np.complex64(2,2)

def test_minimum() -> None:
    block = Minimum()
    assert block.process(np.asarray([0,1,2,3])) == 0
    assert block.process(np.asarray([np.complex64(1,1),np.complex64(2,2)])) == np.complex64(1,1)

def test_modulo() -> None:
    block = Modulo(value=5)

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 1
    assert block.process(np.asarray(-1)) == 4
    assert block.process(np.asarray(5)) == 0
    np.testing.assert_array_equal(block.process(np.asarray([1,2,4,6])),
                                  [1,2,4,1])

def test_multiply() -> None:
    block = Multiply(value=5)

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 5
    assert block.process(np.asarray(-1)) == -5
    assert block.process(np.asarray(np.complex64(1,1))) == np.complex64(5,5)
    assert block.process(np.asarray(np.complex64(0,1))) == np.complex64(0,5)
    assert block.process(np.asarray(np.complex64(-1,-1))) == np.complex64(-5,-5)

    np.testing.assert_array_equal(block.process(np.asarray([1,2,3])),
                                  [5,10,15])
    np.testing.assert_array_equal(block.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[5,5,5],[10,10,10],[15,15,15]])


def test_negate() -> None:
    block = Negate()

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == -1
    assert block.process(np.asarray(-1)) == 1
    assert block.process(np.asarray(np.complex64(1,1))) == np.complex64(-1,-1)
    assert block.process(np.asarray(np.complex64(0,1))) == np.complex64(0,-1)
    assert block.process(np.asarray(np.complex64(-1,-1))) == np.complex64(1,1)
    np.testing.assert_array_equal(
        block.process(np.asarray([np.complex64(1,1),np.complex64(1,-1),np.complex64(1,0)])),
        [np.complex64(-1,-1),np.complex64(-1,1),np.complex64(-1,0)])


def test_power() -> None:
    block = Power(value=5)

    assert block.process(np.asarray(0)) == 0
    assert block.process(np.asarray(1)) == 1
    assert block.process(np.asarray(-1)) == -1
    assert block.process(np.asarray(np.complex64(1,1))) == np.complex64(-4,-4)
    assert block.process(np.asarray(np.complex64(0,1))) == np.complex64(0,1)
    assert block.process(np.asarray(np.complex64(-1,-1))) == np.complex64(4,4)

    np.testing.assert_array_equal(block.process(np.asarray([1,2,3])), 
                                  [1,32,243])
    np.testing.assert_array_equal(block.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[1,1,1],[32,32,32],[243,243,243]])

def test_subtract() -> None:
    block_real = Subtract(value=5)

    assert block_real.process(np.asarray(0)) == -5
    assert block_real.process(np.asarray(1)) == -4
    assert block_real.process(np.asarray(-1)) == -6

    block_complex = Subtract(value=np.complex64(5,5))

    assert block_complex.process(np.asarray(np.complex64(1,1))) == np.complex64(-4,-4)
    assert block_complex.process(np.asarray(np.complex64(0,1))) == np.complex64(-5,-4)
    assert block_complex.process(np.asarray(np.complex64(-1,-1))) == np.complex64(-6,-6)

    block_multiple_values= Subtract(value=[1,2,3])
    np.testing.assert_array_equal(block_multiple_values.process(np.asarray([1,2,3])),
                                  [0,0,0])
    np.testing.assert_array_equal(block_multiple_values.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[0,-1,-2],[1,0,-1],[2,1,0]]
    )

    block_multiple_values2= Subtract(value=[[1],[2],[3]])
    np.testing.assert_array_equal(block_multiple_values2.process(np.asarray([[1,1,1],[2,2,2],[3,3,3]])),
                                  [[0,0,0],[0,0,0],[0,0,0]])

