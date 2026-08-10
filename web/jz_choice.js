// jz Choice — swaps the `choice` text widget for a real combo (dropdown)
// when the `choices` input is wired to a node holding a literal string
// (primitive / string-literal / multiline text nodes). The option list is
// re-read every time the dropdown opens, so upstream edits show up live.
// If the upstream text can't be resolved (computed at runtime), the widget
// stays a plain text field and the server validates the pick instead.

import { app } from "../../scripts/app.js";

const SEPARATORS = { newline: "\n", comma: ",", semicolon: ";", pipe: "|" };

// the connected node's literal text, if it has one
function upstreamText(node) {
  const input = node.inputs?.find((i) => i.name === "choices");
  if (!input || input.link == null) return null;
  const getLink = (id) => node.graph?.links?.get?.(id) ?? node.graph?.links?.[id];
  let link = getLink(input.link);
  let origin = link ? node.graph.getNodeById(link.origin_id) : null;
  // follow simple pass-through nodes (reroutes)
  for (let hop = 0; hop < 5 && origin; hop++) {
    const w = origin.widgets?.find(
      (w) => typeof w.value === "string" && (w.type === "text" || w.type === "customtext" || w.multiline)
    ) ?? origin.widgets?.find((w) => typeof w.value === "string");
    if (w) return w.value;
    const up = origin.inputs?.[0];
    if (!up || up.link == null) break;
    link = getLink(up.link);
    origin = link ? node.graph.getNodeById(link.origin_id) : null;
  }
  return null;
}

function parseChoices(node, text) {
  const sepName = node.widgets?.find((w) => w.name === "separator")?.value ?? "newline";
  return text
    .split(SEPARATORS[sepName] ?? "\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

app.registerExtension({
  name: "jz.Choice",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "jz_Choice") return;

    // replace the text widget with a fresh combo widget (mutating .type in
    // place does not re-render on every frontend version)
    const toCombo = function () {
      const idx = this.widgets.findIndex((w) => w.name === "choice");
      if (idx === -1 || this.widgets[idx].type === "combo") return;
      const old = this.widgets[idx];
      this.__jz_textWidget = old;
      const valuesFn = () => {
        const t = upstreamText(this);
        return t !== null ? parseChoices(this, t) : [];
      };
      const combo = this.addWidget("combo", "choice", old.value, () => {}, { values: valuesFn });
      this.widgets.pop(); // addWidget appended it
      this.widgets.splice(idx, 1, combo);
      this.setDirtyCanvas(true, true);
    };

    const toText = function () {
      const idx = this.widgets.findIndex((w) => w.name === "choice");
      if (idx === -1 || this.widgets[idx].type !== "combo" || !this.__jz_textWidget) return;
      this.__jz_textWidget.value = this.widgets[idx].value;
      this.widgets.splice(idx, 1, this.__jz_textWidget);
      this.setDirtyCanvas(true, true);
    };

    const refresh = function () {
      const text = upstreamText(this);
      if (text !== null && parseChoices(this, text).length) toCombo.call(this);
      else toText.call(this);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      // upstream node may not be in the graph yet during load
      setTimeout(() => refresh.call(this), 0);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      setTimeout(() => refresh.call(this), 0);
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      onAdded?.apply(this, arguments);
      setTimeout(() => refresh.call(this), 0);
    };
  },
});
