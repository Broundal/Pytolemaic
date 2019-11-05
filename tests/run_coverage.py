import argparse
import os


# this creates coverage report
# pip install coverage first

if __name__ == '__main__':
    os.system('pwd')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="file to run unittests. 'all' will run all test under unittests/",
                        default='all')

    parser.add_argument('--source', help="files to measure coverage on", default='../pytolemaic')
    parser.add_argument('--output', help="where to save you html report", default='./htmlcov')
    args = parser.parse_args()

    if args.test == 'all':
        args.test = 'discover -s ../tests/unittests ../tests/integrative_tests'

    call_args = ['coverage', 'run', '--source={}'.format(args.source), '-m unittest', args.test]

    print(' '.join(call_args))
    os.system(' '.join(call_args))
    os.system('coverage report')

    report_args = ['coverage', 'html']
    report_args += ['-d', args.output]
    os.system(' '.join(report_args))