import argparse
import os


# this creates coverage report
if __name__ == '__main__':
    os.system('pwd')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="file to run unittests. 'all' will run all test under unittests/",
                        default='all')
    parser.add_argument('--measure', help="files to measure coverage on", default='../*')
    parser.add_argument('--source', help="files to measure coverage on", default='../pytolemaic')
    parser.add_argument('--html', default=True, action='store_true',
                        help='html- produces an htmlcov report for each of the source files')
    parser.add_argument('--output', help="where to save you html report", default='./htmlcov')
    args = parser.parse_args()

    if args.test == 'all':
        args.test = 'discover -s ../tests/unittests'

    call_args = ['coverage', 'run', '--source={}'.format(args.source), '--include={}'.format(args.measure), '-m unittest', args.test]

    os.system(' '.join(call_args))
    os.system('coverage report')
    if args.html:
        report_args = ['coverage', 'html']
        report_args += ['-d', args.output]
        os.system(' '.join(report_args))