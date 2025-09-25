import pytest

from arthur_common.tools.duckdb_data_loader import (
    escape_identifier,
    escape_str_literal,
    unescape_identifier,
)


def test_escape_identifier():
    assert escape_identifier('foo"bar') == '"foo""bar"'


def test_escape_str_literal():
    assert escape_str_literal("foo'bar") == "'foo''bar'"


@pytest.mark.parametrize(
    "col_name_escaped, col_name_expected_unescaped",
    [
        ('"standard column"', "standard column"),
        ('"column.with.dots."', "column.with.dots."),
        (
            '"struct_field.with.dots"."struct.nested.with.dots."',
            "struct_field.with.dots.struct.nested.with.dots.",
        ),
        ('"one"."two"."three"', "one.two.three"),
        ('"column""with_escaped.quotes"', 'column"with_escaped.quotes'),
        (
            '"nestedColEscapedQuotes"""."nestedQuote""Here"',
            'nestedColEscapedQuotes".nestedQuote"Here',
        ),
        ('"on.""e"."two."""."""th.ree"', 'on."e.two."."th.ree'),
    ],
)
def test_unescape_identifier(
    col_name_escaped: str, col_name_expected_unescaped: str
) -> None:
    unescaped_identifier = unescape_identifier(col_name_escaped)
    assert unescaped_identifier == col_name_expected_unescaped
