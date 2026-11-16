//! Python's `json.dumps` / `json.loads` and PyYAML's `yaml.safe_dump`, byte for byte, over an
//! insertion-ordered value tree.
//!
//! The Python commands write their results with `json.dumps(obj)` (`", "` / `": "` separators),
//! `json.dumps(obj, indent=n)` and `yaml.safe_dump(obj, sort_keys=False)`. Their output depends on
//! two things `serde_json` does differently: dict order (Python keeps insertion order, a
//! `serde_json::Map` without `preserve_order` sorts) and float formatting (`repr(float)` switches
//! to `1e-05` / `1e+16` notation and always keeps a `.0`). [`PyVal`] keeps the order, and
//! [`float_repr`] reproduces `repr`, so a file written here is the file Python writes.

use anyhow::{bail, Result};
use std::fmt::Write as _;

/// Prefix marking a dict key that is a Python `float` (`{0.5: ...}`): `json.dumps` writes it as the
/// string `"0.5"`, `yaml.safe_dump` as the plain scalar `0.5`. Build such keys with [`float_key`].
pub const FLOAT_KEY: char = '\u{1}';

/// The dict key of a Python `float`, see [`FLOAT_KEY`].
pub fn float_key(x: f64) -> String {
    format!("{FLOAT_KEY}{}", float_repr(x))
}

/// A JSON value the way Python holds it: dicts keep insertion order, ints and floats differ.
#[derive(Clone, Debug, PartialEq)]
pub enum PyVal {
    /// `None` / `null`.
    None,
    /// `True` / `False`.
    Bool(bool),
    /// A Python `int`.
    Int(i64),
    /// A Python `float` (NaN and infinities included, as `json.dumps` allows).
    Float(f64),
    /// A `str`.
    Str(String),
    /// A `list` (or tuple).
    List(Vec<PyVal>),
    /// A `dict`, in insertion order.
    Dict(Vec<(String, PyVal)>),
}

impl PyVal {
    /// An empty dict.
    pub fn dict() -> Self {
        PyVal::Dict(Vec::new())
    }

    /// `d[key] = value`: replaces an existing key in place, otherwise appends.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<PyVal>) {
        let key = key.into();
        let value = value.into();
        if let PyVal::Dict(items) = self {
            if let Some(slot) = items.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                items.push((key, value));
            }
        }
    }

    /// Builder form of [`PyVal::set`].
    pub fn with(mut self, key: impl Into<String>, value: impl Into<PyVal>) -> Self {
        self.set(key, value);
        self
    }

    /// `d.get(key)`.
    pub fn get(&self, key: &str) -> Option<&PyVal> {
        match self {
            PyVal::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Mutable `d[key]`.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut PyVal> {
        match self {
            PyVal::Dict(items) => items.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The number as `f64` (ints too), `None` otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PyVal::Int(v) => Some(*v as f64),
            PyVal::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// The string, `None` otherwise.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyVal::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The list items, `None` otherwise.
    pub fn as_list(&self) -> Option<&[PyVal]> {
        match self {
            PyVal::List(v) => Some(v),
            _ => None,
        }
    }

    /// `Option<f64>` as a float or `None`.
    pub fn opt_f64(v: Option<f64>) -> PyVal {
        v.map_or(PyVal::None, PyVal::Float)
    }

    /// `json.dumps(self)`.
    pub fn dumps(&self) -> String {
        dumps(self, None)
    }

    /// `json.dumps(self, indent=indent)`.
    pub fn dumps_indent(&self, indent: usize) -> String {
        dumps(self, Some(indent))
    }

    /// The equivalent `serde_json::Value` (keys sorted; NaN becomes `null`).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            PyVal::None => serde_json::Value::Null,
            PyVal::Bool(b) => (*b).into(),
            PyVal::Int(v) => (*v).into(),
            PyVal::Float(v) => serde_json::Number::from_f64(*v)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            PyVal::Str(s) => s.clone().into(),
            PyVal::List(v) => v.iter().map(PyVal::to_json).collect(),
            PyVal::Dict(items) => serde_json::Value::Object(
                items.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
            ),
        }
    }

    /// From a `serde_json::Value` (whose objects are already key-sorted).
    pub fn from_json(v: &serde_json::Value) -> PyVal {
        match v {
            serde_json::Value::Null => PyVal::None,
            serde_json::Value::Bool(b) => PyVal::Bool(*b),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => PyVal::Int(i),
                None => PyVal::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(s) => PyVal::Str(s.clone()),
            serde_json::Value::Array(a) => PyVal::List(a.iter().map(PyVal::from_json).collect()),
            serde_json::Value::Object(o) => PyVal::Dict(
                o.iter()
                    .map(|(k, v)| (k.clone(), PyVal::from_json(v)))
                    .collect(),
            ),
        }
    }
}

impl From<bool> for PyVal {
    fn from(v: bool) -> Self {
        PyVal::Bool(v)
    }
}
impl From<i64> for PyVal {
    fn from(v: i64) -> Self {
        PyVal::Int(v)
    }
}
impl From<usize> for PyVal {
    fn from(v: usize) -> Self {
        PyVal::Int(v as i64)
    }
}
impl From<i32> for PyVal {
    fn from(v: i32) -> Self {
        PyVal::Int(i64::from(v))
    }
}
impl From<f64> for PyVal {
    fn from(v: f64) -> Self {
        PyVal::Float(v)
    }
}
impl From<&str> for PyVal {
    fn from(v: &str) -> Self {
        PyVal::Str(v.to_string())
    }
}
impl From<String> for PyVal {
    fn from(v: String) -> Self {
        PyVal::Str(v)
    }
}
impl<T: Into<PyVal>> From<Vec<T>> for PyVal {
    fn from(v: Vec<T>) -> Self {
        PyVal::List(v.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<PyVal>> From<Option<T>> for PyVal {
    fn from(v: Option<T>) -> Self {
        v.map_or(PyVal::None, Into::into)
    }
}

/// `repr(float)`: the shortest round-tripping digits, in fixed notation for decimal exponents
/// `-4 <= e < 16` and as `d.ddde+XX` otherwise; `nan`, `inf`, `-inf` for the specials.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` is the shortest round-trip representation: `d.ddde-5`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp >= 0 {
            let int_len = exp + 1;
            if n <= int_len {
                format!("{digits}{}.0", "0".repeat((int_len - n) as usize))
            } else {
                format!("{}.{}", &digits[..int_len as usize], &digits[int_len as usize..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", exp.abs())
    }
}

/// Python's `round(x, ndigits)` for a float.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        float_repr(x)
    }
}

/// `json.dumps(s)` for a string (`ensure_ascii=True`).
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
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
    out
}

/// `json.dumps(v)` (`indent=None`) or `json.dumps(v, indent=n)`.
pub fn dumps(v: &PyVal, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_json(v, indent, 0, &mut out);
    out
}

fn write_json(v: &PyVal, indent: Option<usize>, level: usize, out: &mut String) {
    match v {
        PyVal::None => out.push_str("null"),
        PyVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        PyVal::Int(i) => {
            let _ = write!(out, "{i}");
        }
        PyVal::Float(x) => out.push_str(&json_float(*x)),
        PyVal::Str(s) => out.push_str(&json_str(s)),
        PyVal::List(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(indent, level + 1, out);
                write_json(item, indent, level + 1, out);
            }
            newline(indent, level, out);
            out.push(']');
        }
        PyVal::Dict(items) => {
            if items.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(indent, level + 1, out);
                out.push_str(&json_str(k.strip_prefix(FLOAT_KEY).unwrap_or(k)));
                out.push_str(": ");
                write_json(item, indent, level + 1, out);
            }
            newline(indent, level, out);
            out.push('}');
        }
    }
}

fn newline(indent: Option<usize>, level: usize, out: &mut String) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

/// `json.loads(text)`, keeping dict order and the int/float distinction (`NaN`, `Infinity` and
/// `-Infinity` are accepted, as Python accepts them).
pub fn loads(text: &str) -> Result<PyVal> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        bail!("extra data at byte {} of a JSON document", p.i);
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<PyVal> {
        let Some(&c) = self.s.get(self.i) else {
            bail!("unexpected end of a JSON document");
        };
        match c {
            b'{' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Ok(PyVal::Dict(items));
                }
                loop {
                    self.ws();
                    let PyVal::Str(key) = self.value()? else {
                        bail!("JSON object keys must be strings (byte {})", self.i);
                    };
                    self.ws();
                    if !self.eat(":") {
                        bail!("expected ':' at byte {}", self.i);
                    }
                    self.ws();
                    let v = self.value()?;
                    if let Some(slot) = items.iter_mut().find(|(k, _): &&mut (String, PyVal)| *k == key)
                    {
                        slot.1 = v;
                    } else {
                        items.push((key, v));
                    }
                    self.ws();
                    if self.eat(",") {
                        continue;
                    }
                    if self.eat("}") {
                        return Ok(PyVal::Dict(items));
                    }
                    bail!("expected ',' or '}}' at byte {}", self.i);
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Ok(PyVal::List(items));
                }
                loop {
                    self.ws();
                    items.push(self.value()?);
                    self.ws();
                    if self.eat(",") {
                        continue;
                    }
                    if self.eat("]") {
                        return Ok(PyVal::List(items));
                    }
                    bail!("expected ',' or ']' at byte {}", self.i);
                }
            }
            b'"' => self.string(),
            b't' if self.eat("true") => Ok(PyVal::Bool(true)),
            b'f' if self.eat("false") => Ok(PyVal::Bool(false)),
            b'n' if self.eat("null") => Ok(PyVal::None),
            b'N' if self.eat("NaN") => Ok(PyVal::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Ok(PyVal::Float(f64::INFINITY)),
            b'-' if self.eat("-Infinity") => Ok(PyVal::Float(f64::NEG_INFINITY)),
            _ => self.number(),
        }
    }

    fn number(&mut self) -> Result<PyVal> {
        let start = self.i;
        let mut float = false;
        while self.i < self.s.len() {
            match self.s[self.i] {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => float = true,
                _ => break,
            }
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i])?;
        if text.is_empty() {
            bail!("invalid JSON value at byte {start}");
        }
        if !float {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(PyVal::Int(v));
            }
        }
        Ok(PyVal::Float(text.parse::<f64>().map_err(|e| {
            anyhow::anyhow!("invalid JSON number {text:?}: {e}")
        })?))
    }

    fn string(&mut self) -> Result<PyVal> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                bail!("unterminated JSON string");
            };
            self.i += 1;
            match c {
                b'"' => return Ok(PyVal::Str(out)),
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        bail!("unterminated JSON escape");
                    };
                    self.i += 1;
                    match e {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = |p: &mut Self| -> Result<u32> {
                                let h = std::str::from_utf8(
                                    p.s.get(p.i..p.i + 4).unwrap_or_default(),
                                )?;
                                p.i += 4;
                                Ok(u32::from_str_radix(h, 16)?)
                            };
                            let mut code = hex(self)?;
                            if (0xD800..0xDC00).contains(&code) && self.eat("\\u") {
                                let low = hex(self)?;
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        other => out.push(other as char),
                    }
                }
                _ => {
                    // Copy one UTF-8 sequence.
                    let start = self.i - 1;
                    let mut end = self.i;
                    while end < self.s.len() && (self.s[end] & 0xC0) == 0x80 {
                        end += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.s[start..end])?);
                    self.i = end;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------- YAML

/// `yaml.safe_dump(v, sort_keys=False)` for the block-style documents the CLI writes: nested
/// mappings indented by two spaces, sequences inside a mapping not indented (PyYAML's default),
/// floats as `repr` with `.inf`/`.nan`, `null` for `None`.
pub fn yaml_dump(v: &PyVal) -> String {
    let mut out = String::new();
    match v {
        PyVal::Dict(items) if !items.is_empty() => yaml_map(items, 0, &mut out),
        PyVal::List(items) if !items.is_empty() => yaml_seq(items, 0, &mut out),
        other => {
            out.push_str(&yaml_scalar(other));
            out.push('\n');
        }
    }
    out
}

fn yaml_map(items: &[(String, PyVal)], level: usize, out: &mut String) {
    for (k, v) in items {
        out.push_str(&" ".repeat(level));
        match k.strip_prefix(FLOAT_KEY) {
            Some(float) => out.push_str(float),
            None => out.push_str(&yaml_str(k)),
        }
        out.push(':');
        match v {
            PyVal::Dict(d) if !d.is_empty() => {
                out.push('\n');
                yaml_map(d, level + 2, out);
            }
            PyVal::List(l) if !l.is_empty() => {
                out.push('\n');
                yaml_seq(l, level, out);
            }
            other => {
                out.push(' ');
                out.push_str(&yaml_scalar(other));
                out.push('\n');
            }
        }
    }
}

fn yaml_seq(items: &[PyVal], level: usize, out: &mut String) {
    for v in items {
        out.push_str(&" ".repeat(level));
        out.push('-');
        match v {
            PyVal::Dict(d) if !d.is_empty() => {
                // PyYAML puts the first key on the dash line.
                let mut inner = String::new();
                yaml_map(d, level + 2, &mut inner);
                out.push(' ');
                out.push_str(inner.trim_start());
            }
            PyVal::List(l) if !l.is_empty() => {
                let mut inner = String::new();
                yaml_seq(l, level + 2, &mut inner);
                out.push(' ');
                out.push_str(inner.trim_start());
            }
            other => {
                out.push(' ');
                out.push_str(&yaml_scalar(other));
                out.push('\n');
            }
        }
    }
}

fn yaml_scalar(v: &PyVal) -> String {
    match v {
        PyVal::None => "null".into(),
        PyVal::Bool(b) => if *b { "true" } else { "false" }.into(),
        PyVal::Int(i) => i.to_string(),
        PyVal::Float(x) => {
            if x.is_nan() {
                ".nan".into()
            } else if x.is_infinite() {
                if *x > 0.0 { ".inf" } else { "-.inf" }.into()
            } else {
                // PyYAML: `repr(x).lower()`, with `.0` inserted before an exponent lacking a dot.
                let r = float_repr(*x);
                if !r.contains('.') && r.contains('e') {
                    r.replacen('e', ".0e", 1)
                } else {
                    r
                }
            }
        }
        PyVal::Str(s) => yaml_str(s),
        PyVal::List(_) => "[]".into(),
        PyVal::Dict(_) => "{}".into(),
    }
}

/// A string as PyYAML writes it: plain when that reads back as the same string, otherwise in
/// single quotes (`'` doubled).
fn yaml_str(s: &str) -> String {
    let resolves_to_other = {
        let l = s.to_ascii_lowercase();
        matches!(
            l.as_str(),
            "" | "~"
                | "null"
                | "true"
                | "false"
                | "yes"
                | "no"
                | "on"
                | "off"
                | "y"
                | "n"
                | ".nan"
                | ".inf"
                | "-.inf"
                | "+.inf"
        ) || s.parse::<f64>().is_ok()
            || s.replace('_', "").parse::<i64>().is_ok()
    };
    let first_ok = s
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '/' | '.' | '('));
    let body_ok = s.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(
                c,
                '_' | '/' | '.' | '-' | ' ' | '(' | ')' | '+' | '=' | ',' | '%' | '<' | '>'
            )
    });
    let plain = !resolves_to_other
        && first_ok
        && body_ok
        && !s.ends_with(' ')
        && !s.contains(" #")
        && !s.contains(": ");
    if plain {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_is_python_repr() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1.234e16, "1.234e+16"),
            (123456789012345.0, "123456789012345.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-2.5, "-2.5"),
            (2.0 / 3.0, "0.6666666666666666"),
            (100.0, "100.0"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v = PyVal::dict()
            .with("b", 1i64)
            .with("a", vec![1.0, 0.5])
            .with("e", PyVal::dict())
            .with("n", PyVal::None)
            .with("s", "é\"x");
        // json.dumps({"b": 1, "a": [1.0, 0.5], "e": {}, "n": None, "s": 'é"x'})
        assert_eq!(
            v.dumps(),
            r#"{"b": 1, "a": [1.0, 0.5], "e": {}, "n": null, "s": "\u00e9\"x"}"#
        );
        assert_eq!(
            v.dumps_indent(2),
            "{\n  \"b\": 1,\n  \"a\": [\n    1.0,\n    0.5\n  ],\n  \"e\": {},\n  \"n\": null,\n  \"s\": \"\\u00e9\\\"x\"\n}"
        );
        assert_eq!(loads(&v.dumps()).unwrap(), v);
        assert_eq!(PyVal::Float(f64::NAN).dumps(), "NaN");
    }

    #[test]
    fn yaml_matches_pyyaml_block_style() {
        let v = PyVal::dict()
            .with(
                "measured",
                PyVal::dict()
                    .with("frames", 60i64)
                    .with("grey", vec![1.0, 2.5])
                    .with("d", PyVal::dict().with("x", PyVal::None)),
            )
            .with("psf_px", 1e-5)
            .with("jpeg", vec![40.0, 80.0])
            .with("source", "/tmp/a b/x.mp4")
            .with("flag", "yes");
        assert_eq!(
            yaml_dump(&v),
            "measured:\n  frames: 60\n  grey:\n  - 1.0\n  - 2.5\n  d:\n    x: null\npsf_px: 1.0e-05\n\
             jpeg:\n- 40.0\n- 80.0\nsource: /tmp/a b/x.mp4\nflag: 'yes'\n"
        );
    }

    #[test]
    fn py_round_rounds_like_python() {
        assert_eq!(py_round(2.675, 2), 2.67); // the binary value is below the tie
        assert_eq!(py_round(1.23456, 3), 1.235);
        assert_eq!(py_round(-0.5, 0), -0.0);
    }
}
