import pytest

from tst.memory.keys import CanonicalKeyBuilder, InvalidCanonicalKey


def test_readable_hierarchical_examples():
    keys = CanonicalKeyBuilder(user_id="default", session_id="abc123", project_id="tst")
    assert (
        keys.user_preference("programming_language")
        == "user:default:preference:programming_language"
    )
    assert keys.session_turn(42) == "session:abc123:turn:42"
    assert keys.project_file("router/server.py") == "project:tst:file:router%2Fserver.py"
    assert (
        keys.project_symbol("router.server.run_route")
        == "project:tst:symbol:router.server.run_route"
    )


def test_reserved_prefix_and_traversal_are_rejected():
    with pytest.raises(InvalidCanonicalKey):
        CanonicalKeyBuilder.build("system", "anything")
    with pytest.raises(InvalidCanonicalKey):
        CanonicalKeyBuilder().project_file("../secrets.env")


def test_segments_are_normalized_and_escaped():
    assert CanonicalKeyBuilder.build("User", "A:B C") == "user:a%3Ab_c"
