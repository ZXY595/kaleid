#![forbid(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::undocumented_unsafe_blocks,
    clippy::missing_safety_doc
)]
#![deny(unused_must_use)]
pub mod algorithm;
pub mod frame;
mod utils;
pub mod voxel_map;
