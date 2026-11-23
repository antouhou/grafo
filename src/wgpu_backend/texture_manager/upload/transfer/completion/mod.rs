#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub(super) use browser::TextureUploadCompletion;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub(super) use native::TextureUploadCompletion;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
mod browser;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
mod native;
