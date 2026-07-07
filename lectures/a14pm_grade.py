import unittest
import argparse
import sys
import os

def main():
    # Discover tests in the current directory matching 'tests.py'
    suite = unittest.defaultTestLoader.discover('.', pattern='tests.py')

    # Local execution
    print("Running tests locally...")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Exit with a non-zero status code if tests fail, useful for CI/CD pipelines
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main()
