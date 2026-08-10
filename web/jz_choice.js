// jz Choice — turns the `choice` text widget into a dropdown when the
// `choices` input is wired to a node holding a literal string (primitive /
// string-literal / jz nodes with a multiline items widget). The options are
// re-read every time the dropdown opens, so edits upstream show up live.
// If the upstream text can't be resolved (computed at runtime), the widget
// stays a plain text field and the server validates the pick instead.

import { app } from "../../scripts/app.js";

const SEPARATORS = { newline: "\n", comma: ",", semicolon: ";", pipe: "|" };

// the connected node's literal text, if it has one
function upstreamText(node) {
  const input = node.inputs?.find((i) => i.name === "choices");
  if (!input || input.link == null) return null;
  const link = node.graph?.links?.[input.link];
  if (!link) return null;
  let origin = node.graph.getNodeById(link.origin_id);
  // follow simple reroutes
  for (let hop = 0; hop < 5 && origin; hop++) {
    const w = origin.widgets?.find((w) => typeof w.value === "string");
    if (w) return w.value;
    const up = origin.inputs?.[0];
    if (!up || up.link == null) break;
    const l = node.graph.links?.[up.link];
    origin = l ? node.graph.getNodeById(l.origin_id) : null;
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

    const refresh = function () {
      const w = this.widgets?.find((w) => w.name === "choice");
      if (!w) return;
      if (w.__origType === undefined) w.__origType = w.type;
      const text = upstreamText(this);
      if (text !== null && parseChoices(this, text).length) {
        w.type = "combo";
        w.options = w.options || {};
        // a function: re-evaluated each time the dropdown opens
        w.options.values = () => {
          const t = upstreamText(this);
          return t !== null ? parseChoices(this, t) : [];
        };
      } else {
        w.type = w.__origType;
        if (w.options) delete w.options.values;
      }
      this.setDirtyCanvas(true, true);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      refresh.call(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      // graph links are live by now
      setTimeout(() => refresh.call(this), 0);
    };
  },
});
