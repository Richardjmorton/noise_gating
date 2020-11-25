import unittest
import do_apod3d


class MyTest(unittest.TestCase):

    def test_matmul(self):
        # tests matrix multiplication is working
        self.assertEqual(do_apod3d.do_apod3d(20, 20, 20).shape, (20, 20, 20))
        self.assertEqual(do_apod3d.do_apod3d(20, 20, 10).shape, (20, 20, 10))
        self.assertEqual(do_apod3d.do_apod3d(10, 20, 10).shape, (10, 20, 10))
        self.assertEqual(do_apod3d.do_apod3d(10, 10, 20).shape, (10, 10, 20))


if __name__ == '__main__':
    unittest.main()
