# bedrock-cost

[![crates.io](https://img.shields.io/crates/v/bedrock-cost.svg)](https://crates.io/crates/bedrock-cost)
[![docs.rs](https://img.shields.io/docsrs/bedrock-cost)](https://docs.rs/bedrock-cost)

AWS Bedrock invocation cost across vendors (Llama, Mistral, Cohere,
Titan, AI21). Cross-region inference profile aware (`us.`, `eu.`,
`apac.`). Zero deps.

For **Anthropic Claude** models on Bedrock, use
[`claude-cost`](https://crates.io/crates/claude-cost) — it handles the
extra cache_creation / cache_read fields.

## Usage

```rust
use bedrock_cost::{Usage, default_pricing};

let p = default_pricing("meta.llama3-70b-instruct-v1:0").unwrap();
let u = Usage { input_tokens: 1_000_000, output_tokens: 500_000 };
let cost_usd = p.cost_for(&u);
```

## Model id normalization

```rust
use bedrock_cost::default_pricing;
// Cross-region inference profile
assert_eq!(
    default_pricing("us.meta.llama3-70b-instruct-v1:0"),
    default_pricing("meta.llama3-70b-instruct-v1:0")
);
// Full ARN
assert!(default_pricing("arn:aws:bedrock:us-east-1::foundation-model/cohere.command-r-plus-v1:0").is_some());
```

## Pricing notes

All rates are USD per 1,000,000 tokens (us-east-1 region) as of 2026-Q2.
**Bedrock prices vary by region — verify against
<https://aws.amazon.com/bedrock/pricing/> before billing.**

## License

MIT or Apache-2.0.
