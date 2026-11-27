//! `project.xml` as the token stream `QXmlStreamReader` delivers.
//!
//! The C++ walks the document with `readNext()` and nested `while` loops
//! (`ui/ProjectFileIO.cpp`, `ProjectFileIO::readProjectFile`). The loader here walks a
//! token list in the same way, so element order and nesting are interpreted identically.
//! Element names are local names, as `QXmlStreamReader::name()` returns them; attribute
//! values are normalised as XML 1.0 requires (line ends and tabs become spaces before
//! entities are resolved), as `QXmlStreamReader` does.

use quick_xml::XmlVersion;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

/// One token.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Start {
        name: String,
        attrs: Vec<(String, String)>,
        /// Byte offset of the tag in the document.
        offset: usize,
    },
    End {
        name: String,
    },
    /// Character data, with references resolved.
    Text(String),
}

impl Token {
    /// The local name of a start or end tag.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Token::Start { name, .. } | Token::End { name } => Some(name),
            Token::Text(_) => None,
        }
    }

    #[must_use]
    pub fn is_start(&self, n: &str) -> bool {
        matches!(self, Token::Start { name, .. } if name == n)
    }

    #[must_use]
    pub fn is_end(&self, n: &str) -> bool {
        matches!(self, Token::End { name } if name == n)
    }

    /// `QXmlStreamAttributes::value(name).toString()`: the value, or empty when absent.
    #[must_use]
    pub fn attr(&self, key: &str) -> &str {
        match self {
            Token::Start { attrs, .. } => attrs
                .iter()
                .find(|(k, _)| k == key)
                .map_or("", |(_, v)| v.as_str()),
            _ => "",
        }
    }
}

/// A well-formedness error and where it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XmlError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

/// 1-based line and column (in characters) of a byte offset.
#[must_use]
pub fn line_column(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let mut cut = offset;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let before = &text[..cut];
    let line = before.matches('\n').count() + 1;
    let column = before.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
    (line, column)
}

fn local(name: &str) -> String {
    name.rsplit(':').next().unwrap_or(name).to_owned()
}

/// Tokenises a document. Fails, with a position, where `QXmlStreamReader` would report an
/// error: malformed markup, mismatched or unclosed elements, duplicate attributes, invalid
/// references, or content outside a single root element.
pub fn tokenize(text: &str) -> Result<Vec<Token>, XmlError> {
    let mut tokens = Vec::new();
    tokenize_into(text, &mut tokens, usize::MAX)?;
    Ok(tokens)
}

/// [`tokenize`] refusing a document of more than `max_tokens` tokens (start tags, end tags
/// and texts) with an error at that point, so that hostile input cannot hold more than that
/// in memory (the XMA Portal metadata, ledger F78).
pub fn tokenize_limited(text: &str, max_tokens: usize) -> Result<Vec<Token>, XmlError> {
    let mut tokens = Vec::new();
    tokenize_into(text, &mut tokens, max_tokens)?;
    Ok(tokens)
}

/// The tokens before the first error, and the error: what a loop `while (!xml.atEnd() &&
/// !xml.hasError()) xml.readNext();` sees (`QXmlStreamReader` delivers every token before
/// it reports the error). Used for the XMA Portal metadata a project holds, which XMALab 3
/// can hold twice over (ledger F60).
#[must_use]
pub fn tokenize_lenient(text: &str) -> (Vec<Token>, Option<XmlError>) {
    tokenize_lenient_limited(text, usize::MAX)
}

/// [`tokenize_lenient`] that stops (with an error) after `max_tokens` tokens.
#[must_use]
pub fn tokenize_lenient_limited(text: &str, max_tokens: usize) -> (Vec<Token>, Option<XmlError>) {
    let mut tokens = Vec::new();
    let e = tokenize_into(text, &mut tokens, max_tokens).err();
    (tokens, e)
}

fn tokenize_into(text: &str, tokens: &mut Vec<Token>, max_tokens: usize) -> Result<(), XmlError> {
    let text_no_bom = text.strip_prefix('\u{feff}').unwrap_or(text);
    let bom = text.len() - text_no_bom.len();
    let mut reader = Reader::from_str(text_no_bom);
    reader.config_mut().expand_empty_elements = true;
    reader.config_mut().check_end_names = true;
    let err = |offset: u64, message: String| {
        let (line, column) = line_column(text, bom + usize::try_from(offset).unwrap_or(usize::MAX));
        XmlError {
            message,
            line,
            column,
        }
    };
    let mut depth = 0usize;
    let mut roots = 0usize;
    let mut version = XmlVersion::Implicit1_0;
    loop {
        let before = reader.buffer_position();
        if tokens.len() > max_tokens {
            return Err(err(
                before,
                format!("more than {max_tokens} elements and texts"),
            ));
        }
        let event = reader
            .read_event()
            .map_err(|e| err(reader.error_position(), e.to_string()))?;
        match event {
            Event::Start(e) => {
                if depth == 0 {
                    roots += 1;
                    if roots > 1 {
                        return Err(err(before, "extra content at end of document".into()));
                    }
                }
                depth += 1;
                let mut attrs = Vec::new();
                for a in e.attributes() {
                    let a = a.map_err(|x| err(before, x.to_string()))?;
                    let value = a
                        .normalized_value(version)
                        .map_err(|x| err(before, x.to_string()))?;
                    attrs.push((local(a.key.as_ref()), value.into_owned()));
                }
                tokens.push(Token::Start {
                    name: local(e.name().as_ref()),
                    attrs,
                    offset: bom + usize::try_from(before).unwrap_or(usize::MAX),
                });
            }
            Event::End(e) => {
                depth = depth.saturating_sub(1);
                tokens.push(Token::End {
                    name: local(e.name().as_ref()),
                });
            }
            Event::Text(t) => {
                let s: &str = &t;
                if depth == 0 {
                    if !s.trim().is_empty() {
                        return Err(err(before, "text outside the root element".into()));
                    }
                } else {
                    tokens.push(Token::Text(t.xml_content(version).into_owned()));
                }
            }
            Event::CData(t) => {
                if depth == 0 {
                    return Err(err(before, "CDATA outside the root element".into()));
                }
                let s: &str = &t;
                tokens.push(Token::Text(s.to_owned()));
            }
            Event::GeneralRef(r) => {
                if depth == 0 {
                    return Err(err(before, "reference outside the root element".into()));
                }
                let resolved = match r
                    .resolve_char_ref()
                    .map_err(|x| err(before, x.to_string()))?
                {
                    Some(c) => c.to_string(),
                    None => {
                        let name: &str = &r;
                        match name {
                            "lt" => "<".into(),
                            "gt" => ">".into(),
                            "amp" => "&".into(),
                            "apos" => "'".into(),
                            "quot" => "\"".into(),
                            other => {
                                return Err(err(before, format!("unknown entity '&{other};'")));
                            }
                        }
                    }
                };
                tokens.push(Token::Text(resolved));
            }
            Event::Decl(d) => {
                if let Ok(v) = d.version() {
                    let v: &str = v.as_ref();
                    version = if v == "1.1" {
                        XmlVersion::Explicit1_1
                    } else {
                        XmlVersion::Explicit1_0
                    };
                }
            }
            Event::Eof => {
                if tokens.len() > max_tokens {
                    return Err(err(
                        reader.buffer_position(),
                        format!("more than {max_tokens} elements and texts"),
                    ));
                }
                if depth != 0 {
                    return Err(err(
                        reader.buffer_position(),
                        "premature end of document".into(),
                    ));
                }
                if roots == 0 {
                    return Err(err(
                        reader.buffer_position(),
                        "document has no root element".into(),
                    ));
                }
                return Ok(());
            }
            Event::Empty(_) | Event::Comment(_) | Event::PI(_) | Event::DocType(_) => {}
        }
    }
}

/// `QXmlStreamReader::readElementText()` from a start token at `index`, with its default
/// behaviour `ErrorOnUnexpectedElement` (`qxmlstream.h`: `readElementText(
/// ReadElementTextBehaviour behaviour = ErrorOnUnexpectedElement)`): the text up to the
/// matching end tag; at a child element Qt raises an error and returns what was read so far
/// (Qt's documentation of the enum; the reader then reports `hasError()`, which ends
/// XMALab's reading loops). Returns the text and whether it ended at its end tag (false: a
/// child element, or the end of the tokens). Reads only the tokens up to the first child,
/// so it is linear over a document however its elements nest.
#[must_use]
pub fn read_element_text(tokens: &[Token], index: usize) -> (String, bool) {
    let mut out = String::new();
    for t in tokens.iter().skip(index + 1) {
        match t {
            Token::Start { .. } => return (out, false),
            Token::End { .. } => return (out, true),
            Token::Text(s) => out.push_str(s),
        }
    }
    (out, false)
}

/// The text of [`read_element_text`] (what XMALab's `readElementText()` returns).
#[must_use]
pub fn element_text(tokens: &[Token], index: usize) -> String {
    read_element_text(tokens, index).0
}

// ---------------------------------------------------------------------------------------
// Writing

/// The subset of `QXmlStreamWriter` (Qt 6) that XMALab uses, with auto-formatting on:
/// `writeStartDocument`, `writeStartElement`, `writeAttribute`, `writeCharacters`,
/// `writeEndElement` and `writeEndDocument`.
///
/// Output is byte-identical to Qt's: the declaration `<?xml version="1.0"
/// encoding="UTF-8"?>`; each start tag on a new line, indented by four spaces per level;
/// an element with no content closed as `<X/>`; an end tag on its own line only after child
/// elements (`<X>text</X>` stays on one line); a newline after the root element.
///
/// Escaping (Qt's `QXmlStreamWriterPrivate::writeEscaped`): `<`, `>`, `&` and `"` become
/// `&lt;`, `&gt;`, `&amp;` and `&quot;` everywhere; in attribute values tab, line feed and
/// carriage return become `&#9;`, `&#10;` and `&#13;`; other characters below U+0020, and
/// U+FFFE and U+FFFF, are not allowed in XML and are dropped (Qt drops them too and sets
/// its error flag, which XMALab does not check). Checked against Qt 6.11.
///
/// Ported from Qt 6 `qxmlstream.cpp`, `QXmlStreamWriterPrivate` (`writeStartElement`,
/// `finishStartElement`, `writeEndElement`, `indent`, `writeEscaped`).
#[derive(Debug, Default)]
pub struct XmlWriter {
    out: String,
    stack: Vec<String>,
    in_start_element: bool,
    last_was_start_element: bool,
    wrote_something: bool,
    did_write_start_document: bool,
    did_write_any_token: bool,
}

impl XmlWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `writeStartDocument()`.
    pub fn start_document(&mut self) {
        self.finish_start_element(false);
        self.out
            .push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        self.did_write_start_document = true;
    }

    /// `writeStartElement(name)`.
    pub fn start_element(&mut self, name: &str) {
        if !self.finish_start_element(false) {
            self.indent(self.stack.len());
        }
        self.out.push('<');
        self.out.push_str(name);
        self.stack.push(name.to_owned());
        self.in_start_element = true;
        self.last_was_start_element = true;
        self.did_write_any_token = true;
    }

    /// `writeAttribute(name, value)`. Only valid right after [`XmlWriter::start_element`]
    /// or another attribute, as in Qt.
    pub fn attribute(&mut self, name: &str, value: &str) {
        debug_assert!(self.in_start_element, "attribute outside a start tag");
        self.out.push(' ');
        self.out.push_str(name);
        self.out.push_str("=\"");
        escape_into(&mut self.out, value, true);
        self.out.push('"');
    }

    /// `writeCharacters(text)`.
    pub fn characters(&mut self, text: &str) {
        self.finish_start_element(true);
        escape_into(&mut self.out, text, false);
        self.did_write_any_token = true;
    }

    /// `writeEndElement()`.
    pub fn end_element(&mut self) {
        let Some(name) = self.stack.pop() else {
            return;
        };
        if self.in_start_element {
            self.out.push_str("/>");
            self.in_start_element = false;
            self.last_was_start_element = false;
            return;
        }
        if !self.finish_start_element(false) && !self.last_was_start_element {
            self.indent(self.stack.len());
        }
        self.out.push_str("</");
        self.out.push_str(&name);
        self.out.push('>');
        self.last_was_start_element = false;
    }

    /// `writeEndDocument()`: closes every open element and ends with a newline.
    #[must_use]
    pub fn end_document(mut self) -> String {
        while !self.stack.is_empty() {
            self.end_element();
        }
        self.out.push('\n');
        self.out
    }

    /// Closes a pending start tag. Returns whether content was written since the last
    /// call that passed `contents = false`.
    fn finish_start_element(&mut self, contents: bool) -> bool {
        let had_something_written = self.wrote_something;
        self.wrote_something = contents;
        if self.in_start_element {
            self.out.push('>');
            self.in_start_element = false;
        }
        had_something_written
    }

    fn indent(&mut self, level: usize) {
        if self.did_write_start_document || self.did_write_any_token {
            self.out.push('\n');
        }
        for _ in 0..level {
            self.out.push_str("    ");
        }
    }
}

/// Appends `text` escaped as `QXmlStreamWriter` escapes it (see [`XmlWriter`]).
fn escape_into(out: &mut String, text: &str, attribute: bool) {
    for c in text.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\t' if attribute => out.push_str("&#9;"),
            '\n' if attribute => out.push_str("&#10;"),
            '\r' if attribute => out.push_str("&#13;"),
            '\t' | '\n' | '\r' => out.push(c),
            c if c < ' ' || c == '\u{fffe}' || c == '\u{ffff}' => {}
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qt 6.11 output for the same calls, captured with a small Qt program.
    #[test]
    fn writer_matches_qt() {
        let mut w = XmlWriter::new();
        w.start_document();
        w.start_element("Project");
        w.attribute("A", "<>&\"'\t\n\r x\u{1}y\u{b}z\u{c} w \u{e9} \u{fffe}");
        w.start_element("Empty");
        w.end_element();
        w.start_element("Inner");
        w.start_element("Leaf");
        w.attribute("x", "1");
        w.end_element();
        w.end_element();
        let text = w.end_document();
        assert_eq!(
            text,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <Project A=\"&lt;&gt;&amp;&quot;'&#9;&#10;&#13; xyz w \u{e9} \">\n    \
             <Empty/>\n    <Inner>\n        <Leaf x=\"1\"/>\n    </Inner>\n</Project>\n"
        );

        let mut w = XmlWriter::new();
        w.start_document();
        w.start_element("P");
        assert_eq!(
            w.end_document(),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<P/>\n"
        );
    }

    #[test]
    fn writer_characters_stay_on_the_line() {
        let mut w = XmlWriter::new();
        w.start_document();
        w.start_element("data");
        w.start_element("v");
        w.characters("a\tb<");
        w.end_element();
        w.start_element("e");
        w.characters("");
        w.end_element();
        assert_eq!(
            w.end_document(),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<data>\n    <v>a\tb&lt;</v>\n    <e></e>\n</data>\n"
        );
    }

    #[test]
    fn tokens_and_attributes() {
        let doc = "<?xml version=\"1.0\"?>\n<P a=\"x&amp;y\" b=\"1\n2\"><C n=\"&#65;\"/>t&lt;</P>";
        let t = tokenize(doc).expect("parses");
        assert!(t[0].is_start("P"));
        assert_eq!(t[0].attr("a"), "x&y");
        assert_eq!(t[0].attr("b"), "1 2");
        assert_eq!(t[0].attr("missing"), "");
        assert!(t[1].is_start("C"));
        assert_eq!(t[1].attr("n"), "A");
        assert!(t[2].is_end("C"));
        // `readElementText()` (ErrorOnUnexpectedElement) stops at the child `C` with an
        // error, returning what it read before it (nothing here).
        assert_eq!(read_element_text(&t, 0), (String::new(), false));
        let leaf = tokenize("<P>a&amp;<![CDATA[<b>]]></P>").expect("parses");
        assert_eq!(read_element_text(&leaf, 0), ("a&<b>".to_owned(), true));
    }

    #[test]
    fn errors_have_positions() {
        let e = tokenize("<P>\n<C></P>").unwrap_err();
        assert_eq!(e.line, 2);
        let e = tokenize("<P>\n<C>").unwrap_err();
        assert!(e.message.contains("premature"), "{e:?}");
        assert!(tokenize("<P a=\"1\" a=\"2\"/>").is_err());
        assert!(tokenize("<P/><Q/>").is_err());
        assert!(tokenize("").is_err());
    }
}
