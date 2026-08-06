// jz Display JSON — renders the node's last output as a collapsible,
// syntax-highlighted JSON tree inside the node body.
//
// the last payload is stored in node.properties.jz_json so the view
// survives a workflow save/reload without re-running.

import { app } from "../../scripts/app.js";

const COLORS = {
  key: "#7dcfff",
  string: "#9ece6a",
  number: "#ff9e64",
  keyword: "#bb9af7", // true / false / null
  punct: "#565f89",
  meta: "#565f89", // "3 items" hints on collapsed nodes
  error: "#f7768e",
};

function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (text !== undefined) e.textContent = text;
  return e;
}

function span(color, text) {
  return el("span", { color }, text);
}

// one row of the tree; children of objects/arrays are indented + collapsible
function renderValue(value, key, isLast, depth) {
  const row = el("div", { paddingLeft: depth ? "14px" : "0" });
  const line = el("div", { whiteSpace: "pre-wrap", wordBreak: "break-all" });
  row.appendChild(line);

  const comma = isLast ? "" : ",";
  const keyPart = () => {
    if (key === undefined) return [];
    return [span(COLORS.key, JSON.stringify(key)), span(COLORS.punct, ": ")];
  };

  if (Array.isArray(value) || (value !== null && typeof value === "object")) {
    const isArr = Array.isArray(value);
    const entries = isArr ? value.map((v, i) => [i, v]) : Object.entries(value);
    const open = isArr ? "[" : "{";
    const close = isArr ? "]" : "}";

    const toggle = span(COLORS.punct, entries.length ? "▾ " : "  ");
    toggle.style.cursor = entries.length ? "pointer" : "default";
    line.appendChild(toggle);
    keyPart().forEach((p) => line.appendChild(p));
    line.appendChild(span(COLORS.punct, open));
    const hint = span(
      COLORS.meta,
      ` ${entries.length} ${isArr ? (entries.length === 1 ? "item" : "items") : entries.length === 1 ? "key" : "keys"} `
    );
    hint.style.display = "none";
    hint.style.fontStyle = "italic";
    line.appendChild(hint);
    const closeInline = span(COLORS.punct, close + comma);
    closeInline.style.display = "none";
    line.appendChild(closeInline);

    const body = el("div");
    entries.forEach(([k, v], i) =>
      body.appendChild(renderValue(v, isArr ? undefined : k, i === entries.length - 1, depth + 1))
    );
    const closer = el("div", {});
    closer.appendChild(span(COLORS.punct, close + comma));
    body.appendChild(closer);
    row.appendChild(body);

    if (entries.length) {
      let collapsed = false;
      toggle.onclick = (ev) => {
        ev.stopPropagation();
        collapsed = !collapsed;
        toggle.textContent = collapsed ? "▸ " : "▾ ";
        body.style.display = collapsed ? "none" : "";
        hint.style.display = collapsed ? "" : "none";
        closeInline.style.display = collapsed ? "" : "none";
      };
    }
  } else {
    line.appendChild(span(COLORS.punct, "  "));
    keyPart().forEach((p) => line.appendChild(p));
    if (typeof value === "string") line.appendChild(span(COLORS.string, JSON.stringify(value)));
    else if (typeof value === "number") line.appendChild(span(COLORS.number, String(value)));
    else line.appendChild(span(COLORS.keyword, String(value))); // bool / null
    line.appendChild(span(COLORS.punct, comma));
  }
  return row;
}

function render(container, payload) {
  container.innerHTML = "";
  if (!payload) {
    container.appendChild(el("div", { color: COLORS.meta, fontStyle: "italic" }, "run the workflow to see json here"));
    return;
  }
  if (payload.error) {
    const banner = el("div", {
      color: COLORS.error,
      marginBottom: "6px",
      whiteSpace: "pre-wrap",
    }, payload.error);
    container.appendChild(banner);
    container.appendChild(el("div", { whiteSpace: "pre-wrap", wordBreak: "break-all" }, payload.text));
    return;
  }
  let obj;
  try {
    obj = JSON.parse(payload.text);
  } catch {
    container.appendChild(el("div", { whiteSpace: "pre-wrap", wordBreak: "break-all" }, payload.text));
    return;
  }
  container.appendChild(renderValue(obj, undefined, true, 0));
}

app.registerExtension({
  name: "jz.DisplayJSON",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "jz_DisplayJSON") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);

      const wrap = el("div", {
        background: "rgba(0,0,0,0.25)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "6px",
        padding: "8px",
        overflow: "auto",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        fontSize: "12px",
        lineHeight: "1.45",
        userSelect: "text",
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
      });
      // copy button, appears on hover
      const copy = el("button", {
        position: "sticky",
        top: "0",
        float: "right",
        border: "none",
        borderRadius: "4px",
        padding: "2px 8px",
        cursor: "pointer",
        background: "rgba(255,255,255,0.1)",
        color: "#c0caf5",
        fontSize: "11px",
      }, "copy");
      copy.onclick = (ev) => {
        ev.stopPropagation();
        const text = this.properties?.jz_json?.text ?? "";
        navigator.clipboard?.writeText(text);
        copy.textContent = "copied";
        setTimeout(() => (copy.textContent = "copy"), 1000);
      };
      const content = el("div");
      wrap.appendChild(copy);
      wrap.appendChild(content);
      this._jz_content = content;

      this.addDOMWidget("jz_json_view", "jz_json_view", wrap, {
        serialize: false,
        hideOnZoom: false,
      });
      this.size = [Math.max(this.size[0], 320), Math.max(this.size[1], 220)];
      render(content, this.properties?.jz_json);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const payload = message?.jz_json?.[0];
      if (!payload) return;
      this.properties = this.properties || {};
      this.properties.jz_json = payload; // properties are serialized with the workflow
      if (this._jz_content) render(this._jz_content, payload);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      if (this._jz_content) render(this._jz_content, this.properties?.jz_json);
    };
  },
});
