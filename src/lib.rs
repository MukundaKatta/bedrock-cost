//! # bedrock-cost
//!
//! Calculate AWS Bedrock invocation cost across the non-Anthropic vendors
//! that Bedrock hosts: Meta Llama, Mistral, Cohere, Amazon Titan, and AI21
//! Jurassic. For Anthropic Claude models on Bedrock, use the companion
//! [`claude-cost`](https://crates.io/crates/claude-cost) crate, which
//! handles the cache_creation / cache_read fields Claude exposes.
//!
//! Bedrock IDs are normalized so cross-region inference profile prefixes
//! (`us.`, `eu.`, `apac.`) and ARN paths resolve to the same base model.
//!
//! ## Quick example
//!
//! ```
//! use bedrock_cost::{Usage, default_pricing};
//!
//! let pricing = default_pricing("meta.llama3-70b-instruct-v1:0").unwrap();
//! let usage = Usage { input_tokens: 1_000_000, output_tokens: 500_000 };
//! let cost = pricing.cost_for(&usage);
//! assert!(cost > 0.0);
//! ```
//!
//! ## Cross-region inference profile
//!
//! ```
//! use bedrock_cost::default_pricing;
//! assert_eq!(
//!     default_pricing("us.meta.llama3-70b-instruct-v1:0"),
//!     default_pricing("meta.llama3-70b-instruct-v1:0")
//! );
//! ```
//!
//! Pricing is best-effort and dated; verify against
//! <https://aws.amazon.com/bedrock/pricing/> before using these numbers
//! for billing.

#![deny(missing_docs)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const INFERENCE_PROFILE_PREFIXES: &[&str] = &["us.", "eu.", "apac."];

/// Token usage as returned by Bedrock Converse (`inputTokens` /
/// `outputTokens`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Usage {
    /// Tokens in the input prompt.
    pub input_tokens: u64,
    /// Tokens in the model output.
    pub output_tokens: u64,
}

impl Usage {
    /// Total tokens billed (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Per-model rates, USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pricing {
    /// Fresh input tokens.
    pub input_per_mtok: f64,
    /// Output tokens.
    pub output_per_mtok: f64,
}

impl Pricing {
    /// Compute USD cost for the given usage.
    pub fn cost_for(&self, usage: &Usage) -> f64 {
        (usage.input_tokens as f64 * self.input_per_mtok
            + usage.output_tokens as f64 * self.output_per_mtok)
            / 1_000_000.0
    }
}

/// Strip ARN, inference-profile, and version suffixes from a Bedrock
/// model id.
///
/// `us.meta.llama3-70b-instruct-v1:0` -> `meta.llama3-70b-instruct`
pub fn normalize_model_id(id: &str) -> &str {
    let mut s = id;

    // ARN -> tail after final `/`
    if s.starts_with("arn:aws:bedrock:") {
        if let Some(slash) = s.rfind('/') {
            s = &s[slash + 1..];
        }
    }

    // Cross-region inference-profile prefix
    for prefix in INFERENCE_PROFILE_PREFIXES {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }

    // Trailing `-v\d+:\d+` version suffix
    if let Some(idx) = s.rfind("-v") {
        let tail = &s[idx + 2..];
        if tail
            .splitn(2, ':')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
            && tail.contains(':')
        {
            s = &s[..idx];
        }
    }

    s
}

/// Built-in pricing table. Source: aws.amazon.com/bedrock/pricing as of
/// 2026-Q2 for the us-east-1 region. VERIFY before billing — Bedrock
/// pricing varies by region.
pub const DEFAULT_PRICING_TABLE: &[(&str, Pricing)] = &[
    // Meta Llama 3 family
    (
        "meta.llama3-8b-instruct",
        Pricing {
            input_per_mtok: 0.30,
            output_per_mtok: 0.60,
        },
    ),
    (
        "meta.llama3-70b-instruct",
        Pricing {
            input_per_mtok: 2.65,
            output_per_mtok: 3.50,
        },
    ),
    (
        "meta.llama3-1-8b-instruct",
        Pricing {
            input_per_mtok: 0.22,
            output_per_mtok: 0.22,
        },
    ),
    (
        "meta.llama3-1-70b-instruct",
        Pricing {
            input_per_mtok: 0.72,
            output_per_mtok: 0.72,
        },
    ),
    (
        "meta.llama3-1-405b-instruct",
        Pricing {
            input_per_mtok: 5.32,
            output_per_mtok: 16.00,
        },
    ),
    // Mistral
    (
        "mistral.mistral-large-2407",
        Pricing {
            input_per_mtok: 2.00,
            output_per_mtok: 6.00,
        },
    ),
    (
        "mistral.mistral-small-2402",
        Pricing {
            input_per_mtok: 1.00,
            output_per_mtok: 3.00,
        },
    ),
    // Cohere Command R / R+
    (
        "cohere.command-r-plus",
        Pricing {
            input_per_mtok: 3.00,
            output_per_mtok: 15.00,
        },
    ),
    (
        "cohere.command-r",
        Pricing {
            input_per_mtok: 0.50,
            output_per_mtok: 1.50,
        },
    ),
    // Amazon Titan Text
    (
        "amazon.titan-text-premier",
        Pricing {
            input_per_mtok: 0.50,
            output_per_mtok: 1.50,
        },
    ),
    (
        "amazon.titan-text-express",
        Pricing {
            input_per_mtok: 0.20,
            output_per_mtok: 0.60,
        },
    ),
    (
        "amazon.titan-text-lite",
        Pricing {
            input_per_mtok: 0.15,
            output_per_mtok: 0.20,
        },
    ),
    // AI21 Jamba
    (
        "ai21.jamba-1-5-large",
        Pricing {
            input_per_mtok: 2.00,
            output_per_mtok: 8.00,
        },
    ),
    (
        "ai21.jamba-1-5-mini",
        Pricing {
            input_per_mtok: 0.20,
            output_per_mtok: 0.40,
        },
    ),
];

/// Look up the price table entry for a Bedrock model id.
pub fn default_pricing(model_id: &str) -> Option<Pricing> {
    let key = normalize_model_id(model_id);
    DEFAULT_PRICING_TABLE
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, p)| *p)
}
