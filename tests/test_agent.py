from smart_crop.agent import _parse_decisions


def test_decisions_as_string_is_one_malformed_entry_not_iterated_char_by_char():
    # Real bug found in a full-batch run: the model returned "decisions" as a JSON-encoded string
    # instead of an actual array. Iterating a string directly walks it character by character,
    # logging thousands of single-character "malformed decisions" for one bad response.
    args = {"decisions": '[{"target": "tv", "worthwhile": true}]'}
    plan, malformed = _parse_decisions(args)
    assert plan == {}
    assert len(malformed) == 1


def test_decisions_missing_entirely_is_one_malformed_entry():
    plan, malformed = _parse_decisions({})
    assert plan == {}
    assert len(malformed) == 1


def test_non_dict_element_in_decisions_is_skipped_not_crashed():
    args = {"decisions": [{"target": "tv", "worthwhile": True, "reason": "ok"}, "oops a bare string"]}
    plan, malformed = _parse_decisions(args)
    assert "tv" in plan
    assert malformed == ["oops a bare string"]


def test_valid_decisions_parse_normally():
    args = {
        "decisions": [
            {"target": "tv", "worthwhile": True, "scale": 1.0, "cx": 0.5, "cy": 0.6, "reason": "ok"},
            {"target": "iphone", "worthwhile": False, "reason": "no vertical subject"},
        ]
    }
    plan, malformed = _parse_decisions(args)
    assert malformed == []
    assert plan["tv"][0].cy == 0.6
    assert plan["iphone"][0].worthwhile is False
