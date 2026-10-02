from app.services import safety


def test_detects_chest_pain():
    assert safety.contains_emergency_red_flag("I'm having chest pain during my run")


def test_detects_shortness_of_breath_phrasing():
    assert safety.contains_emergency_red_flag("I can't catch my breath and feel dizzy")


def test_detects_worst_headache():
    assert safety.contains_emergency_red_flag("I just got the worst headache of my life mid-ride")


def test_detects_head_impact_with_warning_signs():
    assert safety.contains_emergency_red_flag(
        "I hit my head on the trail and now I'm vomiting and confused"
    )


def test_detects_calf_swelling_with_warmth():
    assert safety.contains_emergency_red_flag(
        "My calf is swollen and warm and red, one-sided, after a long run"
    )


def test_detects_saddle_numbness():
    assert safety.contains_emergency_red_flag("I have saddle numbness since yesterday's ride")


def test_detects_bowel_bladder_loss():
    assert safety.contains_emergency_red_flag(
        "My back hurts and I'm losing control of my bladder"
    )


def test_detects_severe_worsening_limb_pain():
    assert safety.contains_emergency_red_flag(
        "Severe limb pain that's getting worse and my leg feels numb after I fell"
    )


def test_detects_fever_with_swollen_joint():
    assert safety.contains_emergency_red_flag("I have a fever with a swollen hot knee joint")


def test_ignores_normal_training_questions():
    assert not safety.contains_emergency_red_flag("How much protein do I need after a long ride?")


def test_ignores_routine_soreness():
    assert not safety.contains_emergency_red_flag("My legs are a bit sore after yesterday's tempo run")


def test_ignores_minor_calf_tightness():
    assert not safety.contains_emergency_red_flag("My calf feels a little tight after sprints")


# --- Output-side checks -----------------------------------------------------


def test_looks_like_system_prompt_leak_detects_marker_phrase():
    assert safety.looks_like_system_prompt_leak(
        "Per my medical red flags: instructions, I should tell you to see a doctor."
    )


def test_looks_like_system_prompt_leak_detects_role_description():
    assert safety.looks_like_system_prompt_leak(
        "You are an encouraging, knowledgeable fitness coach, as I was told to say."
    )


def test_looks_like_system_prompt_leak_ignores_normal_reply():
    assert not safety.looks_like_system_prompt_leak(
        "Great question — let's build out a taper week for your race."
    )


def test_reply_missing_safety_deferral_flags_red_flag_without_deferral():
    assert safety.reply_missing_safety_deferral(
        "Chest pain during a run is common, just slow down and keep going easy."
    )


def test_reply_missing_safety_deferral_allows_red_flag_with_deferral():
    assert not safety.reply_missing_safety_deferral(
        "Chest pain during exercise can be serious — please seek emergency medical "
        "attention right away rather than continuing to train."
    )


def test_reply_missing_safety_deferral_ignores_replies_with_no_red_flag():
    assert not safety.reply_missing_safety_deferral(
        "Great job on yesterday's tempo run, your pace looked solid."
    )
