use std::path::Path;

use crate::incremental::Crate;

pub const MANIFEST_FILE_NAME: &str = "ante.toml";

#[derive(serde::Deserialize, Default)]
pub struct Manifest {
    pub name: Option<String>,

    /// Native libraries to link the final binary against, e.g. `link-lib = ["raylib"]`.
    #[serde(rename = "link-lib")]
    pub link_lib: Option<Vec<String>>,

    /// Extra native library search directories, passed to the linker as `-L<path>`.
    #[serde(rename = "link-search")]
    pub link_search: Option<Vec<String>>,
}

impl Manifest {
    /// Read and parse the `ante.toml` manifest in the given crate root directory, if present.
    pub fn read(root: &Path) -> Option<Manifest> {
        let contents = std::fs::read_to_string(root.join(MANIFEST_FILE_NAME)).ok()?;
        toml::from_str(&contents).ok()
    }

    /// Applies this manifest's name & link options onto a dependency crate.
    /// The `name` field overrides the crate's import-visible name.
    pub fn apply(self, crate_: &mut Crate) {
        if let Some(name) = self.name {
            crate_.name = name;
        }
        crate_.link_libs = self.link_lib.unwrap_or_default();
        crate_.link_search_paths = self.link_search.unwrap_or_default();
    }

    /// Applies this manifest's options onto the *local* crate. Unlike [`Manifest::apply`], the
    /// `name` field sets only the output binary name (`output_name`) and does NOT rename the
    /// crate: the local crate keeps its default `Local` import name so `import Local.Foo`
    /// continues to resolve regardless of what the project chose to call its binary.
    pub fn apply_local(self, crate_: &mut Crate) {
        crate_.output_name = self.name;
        crate_.link_libs = self.link_lib.unwrap_or_default();
        crate_.link_search_paths = self.link_search.unwrap_or_default();
    }
}
