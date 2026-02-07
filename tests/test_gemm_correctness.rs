#[cfg(test)]
mod gemm_tests {
    use lele::kernels::gemm;
    use lele::tensor::TensorView;

    #[test]
    fn test_gemm_simple_trans_b() {
        // Test case: A @ B^T + C
        // A: (1, 3) = [[1.0, 2.0, 3.0]]
        // B: (2, 3) = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        // B^T: (3, 2) = [[1, 4], [2, 5], [3, 6]]
        // C: (2,) = [0.5, 1.0]
        // Result: (1, 2)

        let a_data = vec![1.0f32, 2.0, 3.0];
        let a = TensorView::new(&a_data, &[1, 3]);

        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = TensorView::new(&b_data, &[2, 3]);

        let c_data = vec![0.5f32, 1.0];
        let c = TensorView::new(&c_data, &[2]);

        let mut out_buf = Vec::new();
        let result = gemm(&a, &b, Some(&c), 1.0, 1.0, false, true, &mut out_buf);

        // Expected: [1, 2, 3] @ [[1, 4], [2, 5], [3, 6]] + [0.5, 1.0]
        //         = [1*1+2*2+3*3, 1*4+2*5+3*6] + [0.5, 1.0]
        //         = [14, 32] + [0.5, 1.0]
        //         = [14.5, 33.0]

        assert_eq!(result.shape.as_ref(), &[1, 2]);
        assert!(
            (result.data[0] - 14.5).abs() < 1e-5,
            "Expected 14.5, got {}",
            result.data[0]
        );
        assert!(
            (result.data[1] - 33.0).abs() < 1e-5,
            "Expected 33.0, got {}",
            result.data[1]
        );

        println!("✓ Simple Gemm with trans_b test passed");
    }

    #[test]
    fn test_gemm_no_transpose() {
        // Test case: A @ B + C (no transpose)
        // A: (2, 3)
        // B: (3, 2)
        // C: (2,)
        // Result: (2, 2)

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = TensorView::new(&a_data, &[2, 3]);

        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = TensorView::new(&b_data, &[3, 2]);

        let c_data = vec![0.5f32, 1.0];
        let c = TensorView::new(&c_data, &[2]);

        let mut out_buf = Vec::new();
        let result = gemm(&a, &b, Some(&c), 1.0, 1.0, false, false, &mut out_buf);

        // Expected:
        // [[1, 2, 3],    [[1, 2],      [[22, 28],      [[22.5, 29.0],
        //  [4, 5, 6]] @   [3, 4],   =   [49, 64]]   +    [49.5, 65.0]]
        //                 [5, 6]]

        assert_eq!(result.shape.as_ref(), &[2, 2]);
        assert!((result.data[0] - 22.5).abs() < 1e-5);
        assert!((result.data[1] - 29.0).abs() < 1e-5);
        assert!((result.data[2] - 49.5).abs() < 1e-5);
        assert!((result.data[3] - 65.0).abs() < 1e-5);

        println!("✓ Gemm without transpose test passed");
    }

    #[test]
    fn test_gemm_broadcast_bias() {
        // Test different bias broadcasting scenarios
        let a_data = vec![1.0f32, 2.0, 3.0];
        let a = TensorView::new(&a_data, &[1, 3]);

        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = TensorView::new(&b_data, &[2, 3]);

        // Test with scalar bias
        let c_data = vec![10.0f32];
        let c = TensorView::new(&c_data, &[1]);

        let mut out_buf = Vec::new();
        let result = gemm(&a, &b, Some(&c), 1.0, 1.0, false, true, &mut out_buf);

        // Expected: [14, 32] + 10 = [24, 42]
        assert_eq!(result.shape.as_ref(), &[1, 2]);
        assert!((result.data[0] - 24.0).abs() < 1e-5);
        assert!((result.data[1] - 42.0).abs() < 1e-5);

        println!("✓ Gemm with broadcast bias test passed");
    }
}
