use super::bucket::TextureBucket;

#[test]
fn working_set_larger_than_eight_keeps_its_resources_and_acquisition_order() {
    let mut bucket = TextureBucket::default();
    let resources: Vec<_> = (0..24).map(Box::new).collect();
    let pointers: Vec<_> = resources
        .iter()
        .map(|value| &**value as *const u64)
        .collect();
    for resource in resources.into_iter().rev() {
        bucket.recycle(*resource, resource);
    }

    let mut checked_out = Vec::new();
    for _ in 0..3 {
        for (index, pointer) in pointers.iter().enumerate() {
            let resource = bucket.acquire().expect("the whole working set is retained");
            assert_eq!(*resource, index as u64);
            assert_eq!(&*resource as *const u64, *pointer);
            checked_out.push(resource);
        }
        assert!(bucket.acquire().is_none());
        checked_out.rotate_left(7);
        for resource in checked_out.drain(..) {
            bucket.recycle(*resource, resource);
        }
    }
}

#[test]
fn unused_resources_are_released_without_discarding_the_active_working_set() {
    let mut bucket = TextureBucket::default();
    for texture_id in 0..24 {
        bucket.recycle(texture_id, texture_id);
    }
    let active: Vec<_> = (0..10).map(|_| bucket.acquire().unwrap()).collect();
    bucket.discard_unused();
    for texture_id in active.into_iter().rev() {
        bucket.recycle(texture_id, texture_id);
    }
    for texture_id in 0..10 {
        assert_eq!(bucket.acquire(), Some(texture_id));
    }
    assert!(bucket.acquire().is_none());
}
