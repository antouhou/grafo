use super::{TextureUpload, TextureUploadBatch};
use crate::render_backend::texture_manager::{
    TexturePixelLayout, TextureStorageFormat, TextureStreamError,
};
use crate::wgpu_backend::texture_manager::operations::TextureUploadOperation;
use crate::wgpu_backend::WgpuBackendError;
use futures_core::Stream;
use futures_util::StreamExt;
use std::future::Future;
use std::pin::pin;

pub(in super::super) trait TextureBatchWriter {
    fn write(
        &mut self,
        batch: TextureUploadBatch<'_>,
    ) -> impl Future<Output = Result<(), WgpuBackendError>>;
}

pub(in super::super) async fn feed_texture_stream<Source, Chunk, SourceError, Writer>(
    operation: &TextureUploadOperation,
    storage: TextureStorageFormat,
    layout: TexturePixelLayout,
    source: Source,
    writer: &mut Writer,
) -> Result<(), TextureStreamError<WgpuBackendError, SourceError>>
where
    Source: Stream<Item = Result<Chunk, SourceError>>,
    Chunk: AsRef<[u8]>,
    Writer: TextureBatchWriter,
{
    let mut upload = TextureUpload::new(storage, layout)?;
    let mut source = pin!(source);
    while !upload.is_complete() {
        let chunk = operation
            .wait(source.next())
            .await?
            .ok_or_else(|| upload.short_input_error())?
            .map_err(TextureStreamError::Source)?;
        let mut remaining = chunk.as_ref();
        while !remaining.is_empty() && !upload.is_complete() {
            let consumed = upload.convert_into_batch(remaining);
            remaining = &remaining[consumed..];
            if upload.is_batch_ready() {
                operation.wait(writer.write(upload.batch())).await??;
                upload.clear_batch();
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
