//! Ordered JSON written byte for byte the way Python's `json.dumps` writes it.
//!
//! The `label` and `behavior` commands write files (`split.json`, `summary.json`, `naming.json`,
//! `labels.json`) and print JSON that Python code and tests read back, sometimes comparing the
//! text itself (`summary.json == json.dumps(summary, indent=1)`). `serde_json` sorts object keys
//! and prints floats its own way, so this module keeps insertion order and follows
//! `json.dumps`: `", "` / `": "` separators without `indent`, `","` + newline with it, `repr`
//! floats (`1.0`, `1e-05`, `NaN`, `Infinity`) and `\uXXXX` escapes for non-ASCII text.

use std::fmt::Write as _;

/// A JSON value with insertion-ordered objects.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// A Python `int`.
    Int(i64),
    /// A Python `float`.
    Float(f64),
    /// A string.
    Str(String),
    /// A list.
    Arr(Vec<Json>),
    /// A dict, in insertion order.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// An object from `(key, value)` pairs.
    pub fn obj<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Json)>) -> Self {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A string value.
    pub fn str(s: impl Into<String>) -> Self {
        Json::Str(s.into())
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value as a float (`int` or `float`); `None` otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(i) => Some(*i as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The value as a string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `json.dumps(value)` (`indent=None`).
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, 0);
        out
    }

    /// `json.dumps(value, indent=indent)`.
    pub fn dumps_indent(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(indent), 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Json::Float(f) => out.push_str(&float_repr(*f)),
            Json::Str(s) => write_str(out, s),
            Json::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    separator(out, indent, level + 1, i == 0);
                    item.write(out, indent, level + 1);
                }
                close(out, indent, level);
                out.push(']');
            }
            Json::Obj(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    separator(out, indent, level + 1, i == 0);
                    write_str(out, k);
                    out.push_str(": ");
                    v.write(out, indent, level + 1);
                }
                close(out, indent, level);
                out.push('}');
            }
        }
    }

    /// Parses JSON text (through `serde_json`, keeping the key order of the text).
    pub fn parse(text: &str) -> anyhow::Result<Json> {
        // `serde_json::Map` sorts its keys, so the order is recovered from a streaming pass over
        // the text instead: `serde_json::Deserializer` visits map entries in document order.
        use serde::de::{Deserializer as _, MapAccess, SeqAccess, Visitor};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Int(v as i64))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element_seed(Seed)? {
                    items.push(item);
                }
                Ok(Json::Arr(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut pairs = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    pairs.push((key, map.next_value_seed(Seed)?));
                }
                Ok(Json::Obj(pairs))
            }
        }
        struct Seed;
        impl<'de> serde::de::DeserializeSeed<'de> for Seed {
            type Value = Json;
            fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
                d.deserialize_any(V)
            }
        }
        // Python writes `NaN` / `Infinity`, which strict JSON (and serde_json) rejects: they are
        // swapped for sentinels outside string literals and mapped back after parsing.
        let cleaned = replace_specials(text);
        let mut de = serde_json::Deserializer::from_str(&cleaned);
        let value = (&mut de).deserialize_any(V)?;
        de.end()?;
        Ok(restore_specials(value))
    }
}

// Sentinels far outside any value these files hold; matched by range because `serde_json`'s
// float parser is not correctly rounded at this magnitude.
const NAN_SENTINEL: &str = "1.05e308";
const INF_SENTINEL: &str = "1.25e308";
const NEG_INF_SENTINEL: &str = "-1.25e308";

fn replace_specials(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_str = false;
    let mut escaped = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            rest = &rest[c.len_utf8()..];
            continue;
        }
        if c == '"' {
            in_str = true;
        } else if rest.starts_with("NaN") {
            out.push_str(NAN_SENTINEL);
            rest = &rest[3..];
            continue;
        } else if rest.starts_with("-Infinity") {
            out.push_str(NEG_INF_SENTINEL);
            rest = &rest[9..];
            continue;
        } else if rest.starts_with("Infinity") {
            out.push_str(INF_SENTINEL);
            rest = &rest[8..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

fn restore_specials(value: Json) -> Json {
    match value {
        Json::Float(f) if (1.0e308..1.1e308).contains(&f) => Json::Float(f64::NAN),
        Json::Float(f) if (1.2e308..1.3e308).contains(&f) => Json::Float(f64::INFINITY),
        Json::Float(f) if (-1.3e308..-1.2e308).contains(&f) => Json::Float(f64::NEG_INFINITY),
        Json::Arr(items) => Json::Arr(items.into_iter().map(restore_specials).collect()),
        Json::Obj(pairs) => Json::Obj(
            pairs
                .into_iter()
                .map(|(k, v)| (k, restore_specials(v)))
                .collect(),
        ),
        other => other,
    }
}

fn separator(out: &mut String, indent: Option<usize>, level: usize, first: bool) {
    match indent {
        None => {
            if !first {
                out.push_str(", ");
            }
        }
        Some(n) => {
            if !first {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, in positional notation for decimal
/// exponents `-5 < e < 16` and as `1.5e-05` / `1e+16` otherwise.
pub fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // `{:e}` gives the shortest round-trip digits: `d.ddde±x`.
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if (-5 < exp) && (exp < 16) {
        let n = digits.len() as i32;
        let body = if exp >= 0 {
            if n <= exp + 1 {
                format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
            } else {
                let (a, b) = digits.split_at((exp + 1) as usize);
                format!("{a}.{b}")
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let mant = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        let esign = if exp < 0 { "-" } else { "+" };
        format!("{sign}{mant}e{esign}{:02}", exp.abs())
    }
}

/// Python's `f"{v:.{prec}f}"`: like Rust's `{:.prec$}` except for `nan` / `inf` spelling.
pub fn fixed(v: f64, prec: usize) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{v:.prec$}")
    }
}

/// Python's `round(v, digits)` for a float: the nearest decimal with `digits` places, read back.
pub fn py_round(v: f64, digits: usize) -> f64 {
    if !v.is_finite() {
        return v;
    }
    format!("{v:.digits$}").parse().unwrap_or(v)
}

/// Writes `text` to `path`, naming the file in the error.
pub fn write_text(path: &std::path::Path, text: &str) -> anyhow::Result<()> {
    std::fs::write(path, text).map_err(|e| anyhow::anyhow!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_like_python_repr() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456.789, "123456.789"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (-2.5, "-2.5"),
            (0.390, "0.39"),
            (1.5e-7, "1.5e-07"),
            (f64::NAN, "NaN"),
        ];
        for (v, want) in cases {
            assert_eq!(float_repr(v), want, "{v}");
        }
    }

    #[test]
    fn dumps_matches_json_dumps_layout() {
        let v = Json::obj([
            ("a", Json::Int(1)),
            ("b", Json::Arr(vec![Json::Float(0.5), Json::str("x")])),
            ("c", Json::obj(Vec::<(String, Json)>::new())),
        ]);
        assert_eq!(v.dumps(), r#"{"a": 1, "b": [0.5, "x"], "c": {}}"#);
        assert_eq!(
            v.dumps_indent(1),
            "{\n \"a\": 1,\n \"b\": [\n  0.5,\n  \"x\"\n ],\n \"c\": {}\n}"
        );
        assert_eq!(Json::str("é").dumps(), "\"\\u00e9\"");
    }

    #[test]
    fn parse_keeps_key_order_and_specials() {
        let v = Json::parse(r#"{"z": NaN, "a": [1, 2.5, "NaN"], "m": -Infinity}"#).unwrap();
        let Json::Obj(pairs) = &v else { panic!() };
        assert_eq!(
            pairs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["z", "a", "m"]
        );
        assert!(v.get("z").unwrap().as_f64().unwrap().is_nan());
        assert_eq!(v.get("m").unwrap().as_f64(), Some(f64::NEG_INFINITY));
        assert_eq!(
            v.get("a").unwrap(),
            &Json::Arr(vec![Json::Int(1), Json::Float(2.5), Json::str("NaN")])
        );
    }

    #[test]
    fn rounding_and_fixed_follow_python() {
        assert_eq!(py_round(0.16449, 3), 0.164);
        assert_eq!(py_round(2.0, 3), 2.0);
        assert_eq!(fixed(f64::NAN, 2), "nan");
        assert_eq!(fixed(0.125, 2), "0.12");
    }
}
