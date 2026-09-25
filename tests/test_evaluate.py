from src.evaluate import entity_f05, macro_f05


def test_perfect_match():
    score = entity_f05(
        predicted=["S2-001"],
        truth=["S2-001"],
    )

    assert score == 1.0


def test_correct_singleton():
    score = entity_f05(
        predicted=[],
        truth=[],
    )

    assert score == 1.0


def test_wrong_singleton():
    score = entity_f05(
        predicted=["S2-001"],
        truth=[],
    )

    assert score == 0.0


def test_partial_match():
    score = entity_f05(
        predicted=["S2-001", "S2-002"],
        truth=["S2-001"],
    )

    assert 0 < score < 1


def test_macro():
    predictions = {
        "S1-001": ["S2-001"],
        "S1-002": [],
    }

    truth = {
        "S1-001": ["S2-001"],
        "S1-002": [],
    }

    assert macro_f05(predictions, truth) == 1.0