// Shared docx-js helpers for the FraudGuard documents.
const fs = require("fs");
const D = require("docx");
const { Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, ImageRun,
        BorderStyle, Footer, PageNumber, LevelFormat, PageOrientation, TableLayoutType } = D;

const NAVY = "14284B", MUTED = "5B677A", LINE = "D3D9E2", TINT = "EEF2F7", HL = "FFF4D6";
const PORTRAIT_W = 9314, LANDSCAPE_W = 14246; // DXA content widths (A4, 0.9" margins)

function runs(text, base = {}) {
  // **bold**, *italic* inline markup
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g; let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, o.run || {}), spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align, keepNext: o.keepNext });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], spacing: { before: 280, after: 120 }, keepNext: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], spacing: { before: 200, after: 80 }, keepNext: true });
const Bul = (t, level = 0) => new Paragraph({ numbering: { reference: "bul", level }, children: runs(t), spacing: { after: 60, line: 264 } });
const Num = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });
const Caption = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, color: MUTED, size: 18 })], spacing: { after: 200 }, alignment: AlignmentType.LEFT });
const Spacer = () => new Paragraph({ children: [], spacing: { after: 60 } });
const PageBreak = () => new Paragraph({ children: [new D.PageBreak()] });

function Callout(text) {
  return new Paragraph({ children: runs(text), shading: { type: ShadingType.CLEAR, fill: TINT, color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: NAVY, space: 8 } }, spacing: { before: 80, after: 160, line: 276 }, indent: { left: 160, right: 160 } });
}

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function Img(file, widthIn = 6.4, maxHeightIn = 8.6) {
  const { w, h } = pngSize(file); let wPx = widthIn * 96, hPx = wPx * h / w;
  if (hPx > maxHeightIn * 96) { hPx = maxHeightIn * 96; wPx = hPx * w / h; }
  return new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(wPx), height: Math.round(hPx) },
    altText: { title: file.split("/").pop(), description: file.split("/").pop(), name: file.split("/").pop() } })], spacing: { after: 60 }, keepNext: true });
}

function Tbl(headers, rows, widths, o = {}) {
  const size = o.size || 17, total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, i, head, hl) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, color: "auto", fill: head ? NAVY : hl ? HL : "FFFFFF" },
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: (o.right || []).includes(i) && !head ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(txt), { size, color: head ? "FFFFFF" : "1F2A3C", bold: head || (o.boldFirst && i === 0) }) })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
           ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, (o.hl || []).includes(ri))) }))] });
}

function makeDoc(title, subtitle, sections) {
  const footer = (name) => new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
    new TextRun({ text: `FraudGuard · ${name} · page `, size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })] })] });
  const titleBlock = [
    new Paragraph({ children: [new TextRun({ text: "FraudGuard", bold: true, size: 20, color: "B5470A" })], spacing: { after: 40 } }),
    new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(title)], spacing: { after: 60 } }),
    new Paragraph({ children: [new TextRun({ text: subtitle, color: MUTED, size: 20 })], spacing: { after: 240 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: LINE, space: 6 } } }),
  ];
  return new D.Document({
    creator: "FraudGuard team", title, description: subtitle,
    styles: {
      default: { document: { run: { font: "Calibri", size: 21, color: "1F2A3C" } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", run: { size: 44, bold: true, color: NAVY, font: "Calibri" } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: NAVY, font: "Calibri" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, color: NAVY, font: "Calibri" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
      ],
    },
    numbering: { config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } },
                                   { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 260 } } } }] },
      ...["num", "num2", "num3", "num4", "num5"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] })),
    ] },
    sections: sections.map((s, i) => ({
      properties: { page: { size: { width: 11906, height: 16838, orientation: s.landscape ? PageOrientation.LANDSCAPE : PageOrientation.PORTRAIT },
                            margin: { top: 1150, bottom: 1150, left: 1296, right: 1296 } } },
      footers: { default: footer(title) },
      children: i === 0 ? [...titleBlock, ...s.children] : s.children,
    })),
  });
}
async function save(doc, file) { fs.writeFileSync(file, await D.Packer.toBuffer(doc)); console.log("wrote", file); }

const inr = (v) => { v = Number(v); const a = Math.abs(v); if (a >= 1e5) return "₹" + (v / 1e5).toFixed(2) + " L"; return "₹" + Math.round(v).toLocaleString("en-IN"); };
const inrFull = (v) => "₹" + Math.round(Number(v)).toLocaleString("en-IN");
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
function csv(file) {
  const [h, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const split = (l) => { const o = []; let cur = "", q = false; for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { o.push(cur); cur = ""; } else cur += ch; } o.push(cur); return o; };
  const H = split(h); return rows.map((r) => Object.fromEntries(split(r).map((v, i) => [H[i], isNaN(+v) || v === "" ? v : +v])));
}
module.exports = { P, H1, H2, Bul, Num, Caption, Spacer, PageBreak, Callout, Img, Tbl, makeDoc, save, inr, inrFull, pct, csv, PORTRAIT_W, LANDSCAPE_W };
