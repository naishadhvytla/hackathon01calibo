"""Add slide transitions and staggered entrance animations to the FraudGuard deck.

Shapes whose name starts with "anim_" (and charts named "chart_*") fade in automatically,
one after another, when the slide opens. Section dividers push up; content slides fade.
Run after build_deck.js:  python3 add_motion.py ../FraudGuard_Presentation.pptx
"""
import re, sys, zipfile, shutil, os

src = sys.argv[1]
tmp = src + ".tmp"

def timing(spids, step=170, dur=450):
    ids = iter(range(5, 10_000))
    effects = []
    for i, sp in enumerate(spids):
        a, b, c = next(ids), next(ids), next(ids)
        node = "afterEffect" if i == 0 else "withEffect"
        effects.append(
            f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" nodeType="{node}">'
            f'<p:stCondLst><p:cond delay="{i * step}"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
            f'<p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
            f'<p:to><p:strVal val="visible"/></p:to></p:set>'
            f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{c}" dur="{dur}"/><p:tgtEl><p:spTgt spid="{sp}"/></p:tgtEl></p:cBhvr></p:animEffect>'
            f'</p:childTnLst></p:cTn></p:par>')
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            '<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
            '<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            + "".join(effects) +
            '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
            '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
            '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')

with zipfile.ZipFile(src) as zin:
    items = [(i, zin.read(i.filename)) for i in zin.infolist()]
files = {i.filename: d for i, d in items}
section_layouts = {n for n, d in files.items() if n.startswith("ppt/slideLayouts/slideLayout") and n.endswith(".xml") and b'name="SECTION"' in d[:3000]}
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item, data in items:
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", item.filename):
            x = data.decode("utf-8")
            rels = files[item.filename.replace("slides/", "slides/_rels/") + ".rels"].decode("utf-8")
            is_section = any(("ppt/slideLayouts/" + m) in section_layouts for m in re.findall(r"slideLayouts/(slideLayout\d+\.xml)", rels))
            named = re.findall(r'<p:cNvPr id="(\d+)" name="((?:anim|chart)_[A-Za-z0-9_]+)"', x)
            named.sort(key=lambda t: (0 if t[1].startswith("chart") else 1, t[1]))
            trans = '<p:transition spd="slow"><p:push dir="u"/></p:transition>' if is_section else '<p:transition spd="med"><p:fade/></p:transition>'
            block = trans + (timing([sp for sp, _ in named]) if named else "")
            x = x.replace("</p:clrMapOvr>", "</p:clrMapOvr>" + block, 1)
            data = x.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, src)
print("motion added to", src)
