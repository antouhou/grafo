use super::operations::TextureUploadOperation;
use super::upload::stream::{feed_texture_stream, TextureBatchWriter};
use super::upload::{TextureUpload, TextureUploadBatch};
use super::{TextureManagerError, WgpuTextureManager};
use crate::render_backend::texture_manager::{
    TexturePixelLayout, TexturePixels, TextureStorageFormat, TextureStreamError,
};
use crate::wgpu_backend::WgpuBackendError;
use futures_core::Stream;
use wgpu::{CommandEncoderDescriptor, Extent3d, Origin3d, Texture, TextureUsages};

struct WgpuTextureBatchWriter<'a> {
    manager: &'a WgpuTextureManager,
    texture: &'a Texture,
}

impl TextureBatchWriter for WgpuTextureBatchWriter<'_> {
    async fn write(&mut self, batch: TextureUploadBatch<'_>) -> Result<(), WgpuBackendError> {
        self.manager
            .uploads
            .write_async(self.texture, batch)
            .await
            .map_err(WgpuBackendError::from)
    }
}

impl WgpuTextureManager {
    pub(super) fn update_destination(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        size: (u32, u32),
    ) -> Result<(Texture, TextureStorageFormat), WgpuBackendError> {
        self.ensure_available()?;
        let textures = self
            .texture_storage
            .read()
            .expect("texture storage lock poisoned");
        let entry = textures
            .get(&texture_id)
            .ok_or(TextureManagerError::TextureNotFound(texture_id))?;
        let texture_size = (entry.texture.width(), entry.texture.height());
        if u64::from(origin.0) + u64::from(size.0) > u64::from(texture_size.0)
            || u64::from(origin.1) + u64::from(size.1) > u64::from(texture_size.1)
        {
            return Err(TextureManagerError::OutOfBounds {
                origin,
                size,
                texture_size,
            }
            .into());
        }
        Ok((entry.texture.clone(), entry.storage))
    }

    pub(super) fn commit_update(
        &self,
        operation: TextureUploadOperation,
        texture_id: u64,
        destination: &Texture,
        patch: &Texture,
        origin: (u32, u32),
    ) -> Result<(), WgpuBackendError> {
        operation.commit(|| {
            self.ensure_available()?;
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("commit_texture_upload"),
                });
            let mut target = destination.as_image_copy();
            target.origin = Origin3d {
                x: origin.0,
                y: origin.1,
                z: 0,
            };
            encoder.copy_texture_to_texture(
                patch.as_image_copy(),
                target,
                Extent3d {
                    width: patch.width(),
                    height: patch.height(),
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit([encoder.finish()]);
            self.ensure_available()?;
            self.updates.publish(texture_id);
            Ok(())
        })
    }

    pub(super) fn upload_slice(
        &self,
        operation: &TextureUploadOperation,
        texture: &Texture,
        storage: TextureStorageFormat,
        pixels: TexturePixels<'_>,
    ) -> Result<(), WgpuBackendError> {
        let mut remaining = pixels
            .validated_bytes()
            .map_err(TextureManagerError::from)?;
        let mut upload =
            TextureUpload::new(storage, pixels.layout).map_err(TextureManagerError::from)?;
        while !upload.is_complete() {
            operation.check()?;
            let consumed = upload.convert_into_batch(remaining);
            remaining = &remaining[consumed..];
            if upload.is_batch_ready() {
                self.uploads.write(texture, upload.batch())?;
                upload.clear_batch();
            }
        }
        Ok(())
    }

    pub(super) fn update_slice(
        &self,
        texture_id: u64,
        origin: (u32, u32),
        pixels: TexturePixels<'_>,
    ) -> Result<(), WgpuBackendError> {
        self.ensure_available()?;
        let operation = self.operations.begin(texture_id)?;
        let (destination, storage) =
            self.update_destination(texture_id, origin, pixels.layout.size)?;
        pixels
            .validated_bytes()
            .map_err(TextureManagerError::from)?;
        let patch = self.create_texture(pixels.layout.size, storage, TextureUsages::COPY_SRC)?;
        self.upload_slice(&operation, &patch, storage, pixels)?;
        self.commit_update(operation, texture_id, &destination, &patch, origin)
    }

    pub(super) async fn upload_stream<Source, Chunk, SourceError>(
        &self,
        operation: &TextureUploadOperation,
        texture: &Texture,
        storage: TextureStorageFormat,
        layout: TexturePixelLayout,
        source: Source,
    ) -> Result<(), TextureStreamError<WgpuBackendError, SourceError>>
    where
        Source: Stream<Item = Result<Chunk, SourceError>>,
        Chunk: AsRef<[u8]>,
    {
        let mut writer = WgpuTextureBatchWriter {
            manager: self,
            texture,
        };
        feed_texture_stream(operation, storage, layout, source, &mut writer).await
    }
}
