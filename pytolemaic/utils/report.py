import copy


class Report(object):

    def __init__(self, report: dict):

        # flatten to to simple dict
        for k, v in report.items():
            if isinstance(v, Report):
                report[k] = v.report

        self.report = report

    def get(self, key):
        def get_from_report(report, lookup_key):
            if lookup_key in report:
                value = report[lookup_key]
                if isinstance(value, dict):
                    return Report(value)
                else:
                    return report[lookup_key]
            else:
                for key, value in report.items():
                    if isinstance(value, dict):
                        recursion = get_from_report(value, lookup_key)
                        if recursion is not None:
                            return recursion

            # key not found
            return None

        return get_from_report(self.report, lookup_key=key)

    def simplified_keys(self):
        def recursive_replace(report):
            keys = list(report.keys())
            for k in keys:
                v = report[k]
                if str(type(k)).startswith('<enum'):
                    report[k.name] = report.pop(k)
                if isinstance(v, dict):
                    recursive_replace(v)

        report = copy.deepcopy(self.report)
        recursive_replace(report)
        return report

    def __repr__(self):
        return self.simplified_keys().__repr__()
