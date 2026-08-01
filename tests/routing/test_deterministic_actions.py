from tst.routing.deterministic import deterministic_route


def test_persistent_remember_is_store_ltm():
    decision = deterministic_route("Remember that I prefer TypeScript over JavaScript.")
    assert decision.operation == "store"
    assert decision.layer == "ltm"
    assert decision.subject == "programming_language"


def test_session_directive_is_store_stm():
    decision = deterministic_route("For this session, call the service Atlas.")
    assert decision.operation == "store"
    assert decision.layer == "stm"
    assert decision.subject == "service_name"


def test_recent_recall_does_not_mutate():
    decision = deterministic_route("What did I just say?")
    assert decision.operation == "retrieve"
    assert decision.layer == "stm"
    assert not decision.mutates_memory


def test_persistent_recall_does_not_mutate():
    decision = deterministic_route("What did I say about my editor?")
    assert decision.operation == "retrieve"
    assert decision.layer == "ltm"


def test_forget_is_explicit_delete():
    decision = deterministic_route("Forget my editor preference.")
    assert decision.operation == "forget"
    assert decision.subject == "editor"


def test_analyze_command_carries_path():
    decision = deterministic_route("/analyze router/server.py")
    assert decision.operation == "analyze_code"
    assert decision.layer == "tree"
    assert decision.file_path == "router/server.py"


def test_ambiguous_mentions_do_not_trigger_mutation():
    assert deterministic_route("Explain why apps sometimes forget state") is None
    assert deterministic_route("Is 'remember that' a common phrase?") is None


def test_user_facing_retrieval_paraphrases_select_the_expected_scope():
    language = deterministic_route("Which language should we use for the frontend?")
    assert (language.operation, language.layer, language.subject) == (
        "retrieve",
        "ltm",
        "programming_language",
    )
    editor = deterministic_route("Which editor do I use?")
    assert (editor.operation, editor.layer, editor.subject) == ("retrieve", "ltm", "editor")
    service = deterministic_route("What is the service called?")
    assert (service.operation, service.layer, service.subject) == (
        "retrieve",
        "stm",
        "service_name",
    )
