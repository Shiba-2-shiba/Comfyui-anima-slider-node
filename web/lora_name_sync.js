import { app } from "../../scripts/app.js";

const TRAINING_NODES = new Set([
    "ComfyuiAnimaSliderTrainLora",
    "ComfyuiAnimaSliderTrainLoraQpola",
]);
const initializedNodes = new WeakSet();

function promptPrefix(path) {
    const filename = path.trim().split(/[\\/]/).pop();
    const stem = filename.replace(/\.ya?ml$/i, "");
    if (!stem || stem === "." || stem === "..") return null;
    const name = stem.replace(/^prompts-/, "").replace(/^anima-/, "anima_");
    return name ? `loras/${name}` : null;
}

app.registerExtension({
    name: "AnimaSlider.PromptLoraName",
    nodeCreated(node) {
        if (!TRAINING_NODES.has(node.comfyClass ?? node.type) || initializedNodes.has(node)) return;
        const prompt = node.widgets?.find((widget) => widget.name === "prompt_yaml");
        const custom = node.widgets?.find((widget) => widget.name === "custom_prompt_yaml_path");
        const output = node.widgets?.find((widget) => widget.name === "output_lora_prefix");
        if (!prompt || !custom || !output) return;
        initializedNodes.add(node);

        const source = (changedWidget, value) => {
            const customPath = String(changedWidget === custom ? value : custom.value).trim();
            return customPath || String(changedWidget === prompt ? value : prompt.value).trim();
        };
        let lastSource = source();
        let configuring = false;
        const updateName = (path) => {
            const prefix = promptPrefix(path);
            if (prefix !== null) {
                output.value = prefix;
                node.setDirtyCanvas?.(true, true);
            }
        };

        // Creation also precedes workflow loading / cloning. Configure restores
        // serialized widget values; suppress change callbacks during that restore.
        const configure = node.configure;
        node.configure = function (...args) {
            configuring = true;
            try {
                return configure.apply(this, args);
            } finally {
                configuring = false;
                lastSource = source();
            }
        };

        for (const widget of [prompt, custom]) {
            const callback = widget.callback;
            widget.callback = function (value, ...args) {
                const result = callback?.call(this, value, ...args);
                if (!configuring) {
                    const nextSource = source(widget, value);
                    if (nextSource !== lastSource) {
                        lastSource = nextSource;
                        updateName(nextSource);
                    }
                }
                return result;
            };
        }

        // Only initialize the backend defaults; a pre-populated custom name is
        // already user data. Subsequent explicit prompt changes always replace it.
        if (["", "loras/anima_slider", "loras/anima_slider_qpola"].includes(output.value)) {
            updateName(lastSource);
        }
    },
});
