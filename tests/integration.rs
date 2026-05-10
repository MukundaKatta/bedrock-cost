use bedrock_cost::{default_pricing, normalize_model_id, Pricing, Usage};

#[test]
fn basic_llama_cost() {
    let p = default_pricing("meta.llama3-70b-instruct-v1:0").unwrap();
    // 1M input + 500k output: $2.65 + 500k*$3.5/1M = $2.65 + $1.75 = $4.40
    let u = Usage {
        input_tokens: 1_000_000,
        output_tokens: 500_000,
    };
    let cost = p.cost_for(&u);
    assert!((cost - 4.40).abs() < 1e-6, "got {cost}");
}

#[test]
fn cross_region_profile_resolves() {
    assert_eq!(
        default_pricing("us.meta.llama3-70b-instruct-v1:0"),
        default_pricing("meta.llama3-70b-instruct-v1:0")
    );
    assert_eq!(
        default_pricing("eu.mistral.mistral-large-2407-v1:0"),
        default_pricing("mistral.mistral-large-2407")
    );
}

#[test]
fn arn_resolves() {
    let arn = "arn:aws:bedrock:us-east-1::foundation-model/cohere.command-r-plus-v1:0";
    assert_eq!(default_pricing(arn), default_pricing("cohere.command-r-plus"));
}

#[test]
fn unknown_model_is_none() {
    assert!(default_pricing("not-a-real-model").is_none());
}

#[test]
fn normalize_keeps_unversioned() {
    assert_eq!(
        normalize_model_id("amazon.titan-text-lite"),
        "amazon.titan-text-lite"
    );
}

#[test]
fn normalize_strips_only_real_version_suffix() {
    // -v1:0 is a real version suffix.
    assert_eq!(normalize_model_id("foo-v1:0"), "foo");
    // -v1 (no colon) is NOT — keep as-is.
    assert_eq!(normalize_model_id("foo-v1"), "foo-v1");
}

#[test]
fn byo_pricing_works() {
    let p = Pricing {
        input_per_mtok: 1.0,
        output_per_mtok: 2.0,
    };
    let u = Usage {
        input_tokens: 1_000_000,
        output_tokens: 1_000_000,
    };
    assert!((p.cost_for(&u) - 3.0).abs() < 1e-6);
}

#[test]
fn total_tokens_helper() {
    let u = Usage {
        input_tokens: 100,
        output_tokens: 50,
    };
    assert_eq!(u.total_tokens(), 150);
}
