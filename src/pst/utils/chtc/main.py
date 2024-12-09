from docstring_parser import DocstringStyle
from jsonargparse import CLI, set_docstring_parse_options

from pst.utils.chtc import cleanup, job


class Main(job.Main, cleanup.Main):
    pass


def main():
    set_docstring_parse_options(style=DocstringStyle.GOOGLE, attribute_docstrings=True)
    CLI(Main, as_positional=False)


if __name__ == "__main__":
    main()
