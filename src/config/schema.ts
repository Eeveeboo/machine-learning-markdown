export interface MlmdConfig {
  plugins?: string;
  targets?: { lang: string; out: string }[];
}

export function validateConfig(raw: unknown): MlmdConfig | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    console.warn("[mlmd] config: root must be a JSON object");
    return null;
  }
  const obj = raw as Record<string, unknown>;

  if ("plugins" in obj && typeof obj["plugins"] !== "string") {
    console.warn("[mlmd] config: `plugins` must be a string");
    return null;
  }

  if ("targets" in obj) {
    if (!Array.isArray(obj["targets"])) {
      console.warn("[mlmd] config: `targets` must be an array");
      return null;
    }
    for (const t of obj["targets"] as unknown[]) {
      if (
        typeof t !== "object" ||
        t === null ||
        typeof (t as Record<string, unknown>)["lang"] !== "string" ||
        typeof (t as Record<string, unknown>)["out"] !== "string"
      ) {
        console.warn("[mlmd] config: each target must have `lang` and `out` strings");
        return null;
      }
    }
  }

  return raw as MlmdConfig;
}
