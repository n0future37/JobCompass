import pytest

from jobcompass.conditions import SearchRequest, assess
from jobcompass.intent import parse_intent


def test_plain_location_type_and_source_spans():
    request = SearchRequest(query="广州找会计全职，最好有Excel培训")
    parsed = parse_intent(request)
    assert parsed.status == "ready"
    assert parsed.hard.cities == ["广州"] and parsed.hard.employment_types == ["全职"]
    assert parsed.hard.skills_all == []
    assert all(request.query[r.start : r.end] == r.quote for r in parsed.sources if r.source == "query")


def test_explicit_constraints_intersect_and_conflict_never_broadens():
    narrow = parse_intent(SearchRequest(query="只看北京", cities=["北京", "上海"]))
    assert narrow.status == "ready" and narrow.hard.cities == ["北京"]
    conflict = parse_intent(SearchRequest(query="只看广州", cities=["上海"]))
    assert conflict.status == "conflict" and conflict.hard.cities == ["上海"]


def test_history_and_profile_do_not_override_current_conditions():
    parsed = parse_intent(
        SearchRequest(query="之前考虑杭州，现在只看天津的会计实习", profile="过去在北京工作")
    )
    assert parsed.status == "ready" and parsed.hard.cities == ["天津"]
    assert parsed.hard.employment_types == ["实习"]


@pytest.mark.parametrize("word", ["不去", "不在", "不想去", "排除"])
def test_city_negation_never_becomes_positive(word):
    parsed = parse_intent(SearchRequest(query=word + "上海，找会计"))
    assert parsed.status == "ready" and parsed.hard.cities == []
    assert parsed.hard.excluded_cities == ["上海"]


def test_unknown_metadata_cannot_prove_exclusion(job_factory):
    parsed = parse_intent(SearchRequest(query="不要北京，不接受兼职"))
    assert parsed.status == "ready"
    assert not assess(job_factory(cities=None), parsed.conditions())[0]
    assert not assess(job_factory(employment_type_text=None), parsed.conditions())[0]
    assert assess(job_factory(cities=["上海"], employment_type_text="全职"), parsed.conditions())[0]


def test_multicity_source_uses_a_permitted_option(job_factory):
    parsed = parse_intent(SearchRequest(query="北京或上海都可以，不接受北京"))
    assert parsed.status == "ready"
    assert assess(job_factory(cities=["北京", "上海"]), parsed.conditions())[0]
    assert not assess(job_factory(cities=["北京"]), parsed.conditions())[0]


def test_all_included_values_excluded_is_conflict():
    assert parse_intent(SearchRequest(query="只看北京，不去北京")).status == "conflict"


def test_explicit_alias_source_quotes_original_value():
    parsed = parse_intent(SearchRequest(query="找工作", employment_types=["校招"]))
    assert parsed.hard.employment_types == ["校园"]
    assert parsed.sources[0].quote == "校招"


def test_many_natural_city_values_returns_controlled_unsupported():
    from jobcompass.conditions import CITIES

    parsed = parse_intent(SearchRequest(query="、".join(CITIES)))
    assert parsed.status == "unsupported" and len(parsed.hard.cities) == 10


def test_soft_preferences_do_not_become_hard_and_unknown_stays_visible():
    parsed = parse_intent(SearchRequest(query="北京优先，不限城市，双休加分"))
    assert parsed.status == "ready" and parsed.hard.cities == []
    assert parsed.soft_preferences["cities"] == ["北京"]
    assert parse_intent(SearchRequest(query="北京岗位，必须上市公司")).status == "unsupported"


@pytest.mark.parametrize(
    "query,cities,types",
    [
        ("只看北京市或上海市的会计", ["北京", "上海"], []),
        ("只接受全职或者实习", [], ["全职", "实习"]),
    ],
)
def test_supported_or_and_city_suffix(query, cities, types):
    parsed = parse_intent(SearchRequest(query=query))
    assert parsed.status == "ready"
    assert parsed.hard.cities == cities and parsed.hard.employment_types == types


def test_retrieval_enforces_negative_conditions_before_ranking(job_factory):
    from types import SimpleNamespace

    from jobcompass.retrieval import Retriever
    from jobcompass.settings import Settings

    jobs = [
        job_factory("a", jd="负责会计核算", cities=["上海"], employment_type_text="全职"),
        job_factory("b", jd="负责会计核算", cities=["北京"], employment_type_text="全职"),
        job_factory("c", jd="负责会计核算", cities=["上海"], employment_type_text=None),
    ]
    store = SimpleNamespace(
        all_jobs=lambda: jobs, current=lambda keys: {j["job_key"]: j for j in jobs if j["job_key"] in keys}
    )
    result = Retriever(Settings(), store, object()).search(
        SearchRequest(query="找会计，不去北京，不接受兼职", mode="bm25", rerank=False, generate=False)
    )
    assert [r["job"]["job_key"] for r in result["results"]] == ["fixture:a"]
    assert result["intent"]["status"] == "ready" and result["degradations"] == []


def test_conflicting_intent_does_not_return_otherwise_matching_job(job_factory):
    from types import SimpleNamespace

    from jobcompass.retrieval import Retriever
    from jobcompass.settings import Settings

    job = job_factory(cities=["北京"])
    store = SimpleNamespace(all_jobs=lambda: [job])
    result = Retriever(Settings(), store, object()).search(
        SearchRequest(query="只看广州Python", cities=["北京"], generate=False)
    )
    assert result["results"] == [] and result["intent"]["status"] == "conflict"
    assert result["unresolved_conditions"]
