from flip_research.text import extract, norm_ws


def test_extracts_paragraphs_and_drops_chrome(fixture_text):
    ex = extract(fixture_text("article.html"))
    assert ex.title == "Tide Tables & Harbour Notes"
    paras = ex.text.split("\n\n")
    assert paras[0] == "Harbour tides"
    assert paras[1] == ("The harbour at Port Example floods twice a day, and the highest tide "
                        "of the year arrives in March.")
    assert paras[2] == "Fishing boats leave before dawn. Most return by noon, carrying cod & haddock."
    assert "A bare div of text counts as a paragraph too." in paras
    assert "After a line break." in paras
    assert "First list item about moorings" in paras
    for chrome in ("SCRIPT-TEXT", "color: red", "Nav item", "Home page chrome", "sidebar",
                   "footer chrome", "newsletter", "enable JavaScript"):
        assert chrome not in ex.text


def test_malformed_html_does_not_raise():
    ex = extract("<p>unclosed <b>bold <p>next<script>x</p>")
    assert "unclosed bold" in ex.text


def test_empty_input():
    assert extract("").text == ""
    assert extract("<script>only()</script>").text == ""


def test_norm_ws():
    assert norm_ws("  a \n\t b  ") == "a b"
