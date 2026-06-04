use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    pub lang: String,
    pub out: String,
}

/// Flexible plugin path: either a single path string or an array of paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PluginPaths {
    Single(String),
    Multiple(Vec<String>),
}

impl PluginPaths {
    /// Return all plugin paths as a flat `Vec<&str>`.
    pub fn as_paths(&self) -> Vec<&str> {
        match self {
            PluginPaths::Single(p) => vec![p.as_str()],
            PluginPaths::Multiple(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }

    /// Return `true` if no paths are configured.
    pub fn is_empty(&self) -> bool {
        match self {
            PluginPaths::Single(_) => false,
            PluginPaths::Multiple(v) => v.is_empty(),
        }
    }

    /// Return the number of paths.
    pub fn len(&self) -> usize {
        match self {
            PluginPaths::Single(_) => 1,
            PluginPaths::Multiple(v) => v.len(),
        }
    }
}

/// Convenience: convert a single `String` or `Vec<String>` into `PluginPaths`.
impl From<String> for PluginPaths {
    fn from(s: String) -> Self {
        PluginPaths::Single(s)
    }
}
impl From<Vec<String>> for PluginPaths {
    fn from(v: Vec<String>) -> Self {
        PluginPaths::Multiple(v)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlmdConfig {
    /// Plugin path(s): either a single string or an array of strings.
    /// Each entry may be a `.so`/`.dylib` (native) or `.wasm` (WASM) file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<PluginPaths>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<TargetConfig>>,
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

/// Load configuration from the given path, or try `.mlmdrc` in the current
/// or parent directories.
pub fn load_config(path: Option<&str>) -> Result<Option<MlmdConfig>, String> {
    let config_path = match path {
        Some(p) => Path::new(p).to_path_buf(),
        None => find_config()?,
    };

    if !config_path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read config file: {}", e))?;

    let config: MlmdConfig =
        serde_json::from_str(&content).map_err(|e| format!("failed to parse config: {}", e))?;

    Ok(Some(config))
}

/// Look for `.mlmdrc` in the current directory and parent directories.
fn find_config() -> Result<std::path::PathBuf, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("failed to get current directory: {}", e))?;

    let mut dir = Some(cwd.as_path());
    while let Some(d) = dir {
        let candidate = d.join(".mlmdrc");
        if candidate.exists() {
            return Ok(candidate);
        }
        dir = d.parent();
    }

    // Return a path that won't exist — caller handles absense
    Ok(cwd.join(".mlmdrc"))
}

/// Save a configuration to `.mlmdrc`.
///
/// Writes to the nearest existing `.mlmdrc` (walking up the directory tree),
/// or creates a new one in the current working directory if none exists.
pub fn save_config(config: &MlmdConfig) -> Result<(), String> {
    let path = find_config()?;
    let content = serde_json::to_string_pretty(config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    fs::write(&path, &content)
        .map_err(|e| format!("failed to write config to '{}': {e}", path.display()))
}
