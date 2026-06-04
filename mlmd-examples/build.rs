use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // The manifest directory is mlmd-examples/
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let examples_dir = Path::new(&manifest_dir).parent().unwrap().join("examples");

    // Canonicalize to absolute path
    let examples_dir =
        fs::canonicalize(&examples_dir).expect("failed to canonicalize examples directory");

    // Tell cargo to rerun if anything in the examples directory changes
    println!("cargo:rerun-if-changed={}", examples_dir.display());

    // Discover all *.rs files in the examples directory
    let mut model_modules: Vec<String> = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(&examples_dir)
        .expect("failed to read examples directory")
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "rs") {
            let file_stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            let absolute_path =
                fs::canonicalize(&path).expect("failed to canonicalize example file path");
            let mod_name = file_stem.to_lowercase().replace(".", "_");
            model_modules.push(format!(
                "#[cfg(test)]\nmod {mod_name} {{ include!(\"{path}\"); }}\n",
                mod_name = mod_name,
                path = absolute_path.display()
            ));
        }
    }

    // Write the generated module file to OUT_DIR
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("models.rs");
    fs::write(&out_path, model_modules.join("\n")).expect("failed to write models.rs");
}
