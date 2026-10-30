"""Spec from the prompt (plain text in, no fixtures)."""
import shotguard_core as sg

CAST = "Rosa: baker, yellow headscarf\nRival: influencer, gold puffer vest"
PROMPT = ('Animate this start frame. The baker in a yellow headscarf stands on the left behind the counter and says '
          'quietly: "We have been here forty years." The influencer in a gold puffer vest laughs: "Enjoy it while it lasts." '
          'He walks out. A voice off screen shouts: "Next!" The sign above the door reads "MEGABUN". One continuous take.')


def test_lines_speakers_text_and_sides():
    e = sg.prompt_spec.extract(PROMPT, CAST)
    assert [(L["speaker"], L["text"]) for L in e["script"]] == [
        ("Rosa", "We have been here forty years."), ("Rival", "Enjoy it while it lasts."), ("", "Next!")]
    assert e["allowed_text"] == ["MEGABUN"] and e["sides"] == {"Rosa": "left"}


def test_retry_hint_is_not_read_as_script():
    hinted = PROMPT + ' Shot Guard fix: say line 1 word for word: "We have been here forty years.". Keep everything else identical.'
    assert len(sg.prompt_spec.extract(hinted, CAST)["script"]) == 3


def test_script_style_prompt_and_full_spec():
    assert sg.prompt_spec.extract("Rosa: Hello.\nRival: Bye.")["script"] == [{"speaker": "Rosa", "text": "Hello."},
                                                                            {"speaker": "Rival", "text": "Bye."}]
    sp = sg.prompt_spec.make_spec(PROMPT, CAST, allowed_text=["OPEN"], max_retries=1)
    assert [c["name"] for c in sp["characters"]] == ["Rosa", "Rival"] and sp["characters"][0]["side"] == "left"
    assert sp["allowed_text"] == ["OPEN", "MEGABUN"] and sp["max_retries"] == 1
