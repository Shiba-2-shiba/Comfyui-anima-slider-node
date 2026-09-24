import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

// Load the actual browser extension with only ComfyUI's app import substituted.
const source = await readFile(new URL("../web/lora_name_sync.js", import.meta.url), "utf8");
let extension;
const context = vm.createContext();
const appModule = new vm.SyntheticModule(["app"], function () {
    this.setExport("app", { registerExtension(value) { extension = value; } });
}, { context });
const module = new vm.SourceTextModule(source, { context });
await module.link((specifier) => {
    assert.equal(specifier, "../../scripts/app.js");
    return appModule;
});
await module.evaluate();

const AGE = "prompts-anima-aging_slider_fullbody.yaml";
const BREAST = "prompts-anima-breast_size_slider_v4.yaml";
const TYPES = ["ComfyuiAnimaSliderTrainLora", "ComfyuiAnimaSliderTrainLoraQpola"];

function makeNode(type = TYPES[0], output = "loras/anima_slider") {
    return {
        comfyClass: type,
        widgets: [
            { name: "prompt_yaml", value: AGE },
            { name: "custom_prompt_yaml_path", value: "" },
            { name: "output_lora_prefix", value: output },
        ],
        redraws: 0,
        setDirtyCanvas() { this.redraws++; },
        configure(values) {
            // Simulate a loader firing callbacks after restoring *all* values.
            // An unguarded callback would overwrite the saved output name here.
            for (const widget of this.widgets) widget.value = values[widget.name];
            for (const widget of this.widgets) widget.callback?.call(widget, widget.value);
            return "configured";
        },
    };
}

function change(node, name, value) {
    const widget = node.widgets.find((item) => item.name === name);
    widget.value = value;
    return widget.callback?.call(widget, value, "extra-argument");
}

const output = (node) => node.widgets[2].value;

for (const type of TYPES) {
    test(`${type}: initialize, manually edit, then select another prompt`, () => {
        const node = makeNode(type, type.endsWith("Qpola") ? "loras/anima_slider_qpola" : "loras/anima_slider");
        extension.nodeCreated(node);
        assert.equal(output(node), "loras/anima_aging_slider_fullbody");
        change(node, "output_lora_prefix", "loras/anima_aging_slider_fullbody_test01");
        assert.equal(output(node), "loras/anima_aging_slider_fullbody_test01");
        change(node, "prompt_yaml", BREAST);
        assert.equal(output(node), "loras/anima_breast_size_slider_v4");
        assert.ok(node.redraws >= 2);
    });

    test(`${type}: loading / cloning preserves saved names, including old defaults`, () => {
        for (const savedName of ["loras/anima_age_slider", "loras/my_run_test01", "loras/anima_slider", ""]) {
            const node = makeNode(type);
            extension.nodeCreated(node);
            assert.equal(node.configure({
                prompt_yaml: BREAST,
                custom_prompt_yaml_path: "",
                output_lora_prefix: savedName,
            }), "configured");
            assert.equal(output(node), savedName);
            change(node, "prompt_yaml", AGE);
            assert.equal(output(node), "loras/anima_aging_slider_fullbody");
        }
    });
}

test("custom paths take priority; clearing returns to selected YAML", () => {
    const node = makeNode();
    extension.nodeCreated(node);
    change(node, "custom_prompt_yaml_path", " C:\\prompts\\custom_trial.YAML ");
    assert.equal(output(node), "loras/custom_trial");
    change(node, "output_lora_prefix", "loras/custom_trial_test01");
    change(node, "prompt_yaml", BREAST);
    assert.equal(output(node), "loras/custom_trial_test01");
    change(node, "custom_prompt_yaml_path", "/data/prompts/prompts-anima-smile_intensity_slider.yml");
    assert.equal(output(node), "loras/anima_smile_intensity_slider");
    change(node, "custom_prompt_yaml_path", "   ");
    assert.equal(output(node), "loras/anima_breast_size_slider_v4");
});

test("restoring a custom path preserves its saved name and resets change tracking", () => {
    const node = makeNode();
    extension.nodeCreated(node);
    node.configure({ prompt_yaml: BREAST, custom_prompt_yaml_path: "/data/custom.yaml", output_lora_prefix: "loras/saved" });
    change(node, "prompt_yaml", AGE);
    assert.equal(output(node), "loras/saved");
    change(node, "custom_prompt_yaml_path", "");
    assert.equal(output(node), "loras/anima_aging_slider_fullbody");
});

test("same selection and repeated initialization preserve manual edits", () => {
    const node = makeNode();
    extension.nodeCreated(node);
    change(node, "output_lora_prefix", "loras/manual");
    change(node, "prompt_yaml", AGE);
    extension.nodeCreated(node);
    assert.equal(output(node), "loras/manual");
    change(node, "prompt_yaml", BREAST);
    assert.equal(output(node), "loras/anima_breast_size_slider_v4");
});

test("existing callback receives its context and arguments exactly once", () => {
    const node = makeNode();
    const widget = node.widgets[0];
    let calls = 0;
    widget.callback = function (...args) {
        assert.equal(this, widget);
        assert.deepEqual(args, [BREAST, "extra-argument"]);
        calls++;
        return "original-result";
    };
    extension.nodeCreated(node);
    assert.equal(change(node, "prompt_yaml", BREAST), "original-result");
    assert.equal(calls, 1);
});

test("nodes remain independent and unrelated nodes are untouched", () => {
    const first = makeNode();
    const second = makeNode();
    const unrelated = makeNode("AnotherNode");
    const configure = unrelated.configure;
    for (const node of [first, second, unrelated]) extension.nodeCreated(node);
    change(first, "prompt_yaml", BREAST);
    assert.equal(output(second), "loras/anima_aging_slider_fullbody");
    assert.equal(output(unrelated), "loras/anima_slider");
    assert.equal(unrelated.configure, configure);
});

test("pre-populated custom names are preserved on initialization", () => {
    const node = makeNode(TYPES[0], "loras/custom_saved_name");
    extension.nodeCreated(node);
    assert.equal(output(node), "loras/custom_saved_name");
});

test("missing widgets and empty prompt options are tolerated", () => {
    extension.nodeCreated({ comfyClass: TYPES[0] });
    const node = makeNode();
    node.widgets[0].value = "";
    extension.nodeCreated(node);
    assert.equal(output(node), "loras/anima_slider");
    change(node, "custom_prompt_yaml_path", "/data/");
    assert.equal(output(node), "loras/anima_slider");
});
