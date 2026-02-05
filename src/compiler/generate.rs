use super::{Allocator, AnalysisData, Compiler, sanitize_name};
use crate::model::onnx_proto::{NodeProto, ValueInfoProto};
use std::collections::{HashMap, HashSet};
use std::io::Write;

pub(crate) struct OpContext<'a> {
    pub node: &'a NodeProto,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub buf_expr: String,
    pub indent: usize,
    pub known_weights: &'a HashMap<String, (usize, usize, Vec<usize>, i32)>,
    pub int64_map: &'a HashMap<String, (Vec<i64>, Vec<usize>)>,
    #[allow(dead_code)]
    pub allocator: Option<&'a Allocator>,
    #[allow(dead_code)]
    pub analysis: Option<&'a AnalysisData>,
    pub current_id: &'a mut usize,
    pub compiler: &'a Compiler,
    pub var_types: &'a HashMap<String, String>,
}

impl<'a> OpContext<'a> {
    pub fn tab(&self) -> String {
        "    ".repeat(self.indent)
    }
}

pub(crate) fn collect_recursive_metrics(
    nodes: &[&NodeProto],
    defined: &mut HashSet<String>,
    used: &mut HashSet<String>,
) {
    for node in nodes {
        for out in &node.output {
            if !out.is_empty() {
                defined.insert(out.clone());
            }
        }
        for inp in &node.input {
            if !inp.is_empty() {
                used.insert(inp.clone());
            }
        }
        // For subgraphs (If, Loop), recursively collect what they use
        // This is needed because subgraphs can access outer scope variables
        for attr in &node.attribute {
            if let Some(g) = &attr.g {
                let mut sub_defined = HashSet::new();
                let mut sub_used = HashSet::new();
                // Recursively collect from subgraph
                let sub_nodes: Vec<&NodeProto> = g.node.iter().collect();
                collect_recursive_metrics(&sub_nodes, &mut sub_defined, &mut sub_used);
                // What the subgraph uses but doesn't define itself needs to come from outer scope
                for u in &sub_used {
                    if !sub_defined.contains(u) {
                        used.insert(u.clone());
                    }
                }
                // Note: we don't add sub_defined to defined, because those are scoped to the subgraph
            }
        }
    }
}

pub(crate) fn infer_variable_types(
    nodes: &[&NodeProto],
    int64_map: &HashMap<String, (Vec<i64>, Vec<usize>)>,
    graph_inputs: &[ValueInfoProto],
    known_weights: &HashMap<String, (usize, usize, Vec<usize>, i32)>,
) -> HashMap<String, String> {
    fn collect_nodes_recursive<'a>(nodes: &[&'a NodeProto], out: &mut Vec<&'a NodeProto>) {
        for node in nodes {
            out.push(*node);
            for attr in &node.attribute {
                if let Some(g) = &attr.g {
                    let sub_nodes: Vec<&NodeProto> = g.node.iter().collect();
                    collect_nodes_recursive(&sub_nodes, out);
                }
            }
        }
    }

    let mut all_nodes: Vec<&NodeProto> = Vec::new();
    collect_nodes_recursive(nodes, &mut all_nodes);

    let mut producer_map: HashMap<String, &NodeProto> = HashMap::new();
    for node in &all_nodes {
        for out in &node.output {
            if !out.is_empty() {
                producer_map.insert(out.clone(), *node);
            }
        }
    }

    let constant_of_shape_output_type = |node: &NodeProto| -> String {
        let mut out_type = "f32".to_string();
        if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
            if let Some(t) = &attr.t {
                let dt = t.data_type;
                if dt == 6 || dt == 7 {
                    out_type = "i64".to_string();
                } else if dt == 1 || dt == 10 {
                    out_type = "f32".to_string();
                } else if !t.int64_data.is_empty() || !t.int32_data.is_empty() {
                    out_type = "i64".to_string();
                } else if !t.float_data.is_empty() {
                    out_type = "f32".to_string();
                }
            }
        }
        out_type
    };

    let mut types = HashMap::new();
    // Pre-populate with graph inputs
    for input in graph_inputs {
        let name = sanitize_name(&input.name);
        let mut ty_str = "f32".to_string();

        if let Some(ty) = &input.r#type {
            if let Some(val) = &ty.value {
                use crate::model::onnx_proto::type_proto::Value;
                match val {
                    Value::TensorType(tt) => {
                        if tt.elem_type == 7 || tt.elem_type == 6 {
                            ty_str = "i64".to_string();
                        }
                    }
                    _ => {}
                }
            }
        }
        types.insert(name, ty_str);
    }
    // Pre-populate with int64_map constants
    for name in int64_map.keys() {
        types.insert(sanitize_name(name), "i64".to_string());
    }

    // Pre-populate with weights
    for (name, (_, _, _, ty)) in known_weights {
        if *ty == 7 || *ty == 6 {
            types.insert(name.clone(), "i64".to_string());
        } else if *ty == 1 {
            types.insert(name.clone(), "f32".to_string());
        }
    }

    // Propagation pass
    let mut changed = true;
    let mut passes = 0;
    while changed && passes < 100 {
        changed = false;
        passes += 1;
        for node in &all_nodes {
            let op = node.op_type.as_str();

            match op {
                "ConstantOfShape" => {
                    // ConstantOfShape output type depends on the "value" attribute.
                    // Default is float (f32) if not specified.
                    let mut out_type = "f32".to_string();
                    if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                        if let Some(t) = &attr.t {
                            let dt = t.data_type;
                            if dt == 6 || dt == 7 {
                                out_type = "i64".to_string();
                            } else if dt == 1 || dt == 10 {
                                out_type = "f32".to_string();
                            } else if !t.int64_data.is_empty() || !t.int32_data.is_empty() {
                                out_type = "i64".to_string();
                            } else if !t.float_data.is_empty() {
                                out_type = "f32".to_string();
                            }
                        }
                    }
                    for out in &node.output {
                        if !out.is_empty() {
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&out_type) {
                                types.insert(name, out_type.clone());
                                changed = true;
                            }
                        }
                    }
                }
                "Shape" | "Size" | "ArgMax" | "NonZero" => {
                    for out in &node.output {
                        if !out.is_empty() {
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&"i64".to_string()) {
                                types.insert(name, "i64".to_string());
                                changed = true;
                            }
                        }
                    }
                }
                "Greater" | "Less" | "Equal" | "And" | "Or" | "Not" | "GreaterOrEqual"
                | "LessOrEqual" => {
                    for out in &node.output {
                        if !out.is_empty() {
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&"i64".to_string()) {
                                types.insert(name, "i64".to_string());
                                changed = true;
                            }
                        }
                    }
                }
                "Range" => {
                    let mut has_f32 = false;
                    let mut has_i64 = false;
                    for inp in &node.input {
                        if inp.is_empty() {
                            continue;
                        }
                        let name = sanitize_name(inp);
                        if let Some(t) = types.get(&name) {
                            if t == "f32" {
                                has_f32 = true;
                            }
                            if t == "i64" {
                                has_i64 = true;
                            }
                        }
                    }
                    let output_type = if has_f32 {
                        "f32"
                    } else if has_i64 {
                        "i64"
                    } else {
                        "f32"
                    };
                    for inp in &node.input {
                        if inp.is_empty() {
                            continue;
                        }
                        let name = sanitize_name(inp);
                        if types.get(&name) != Some(&output_type.to_string()) {
                            types.insert(name, output_type.to_string());
                            changed = true;
                        }
                    }
                    for out in &node.output {
                        if !out.is_empty() {
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&output_type.to_string()) {
                                types.insert(name, output_type.to_string());
                                changed = true;
                            }
                        }
                    }
                }
                "Cast" => {
                    let to = node.attribute.iter().find(|a| a.name == "to").map(|a| a.i);
                    let output_type = if to == Some(7) || to == Some(6) || to == Some(9) {
                        "i64".to_string()
                    } else {
                        "f32".to_string()
                    };
                    for out in &node.output {
                        if !out.is_empty() {
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&output_type) {
                                types.insert(name, output_type.clone());
                                changed = true;
                            }
                        }
                    }
                }
                "Reshape" | "Unsqueeze" | "Squeeze" | "Slice" | "Flatten" | "Transpose"
                | "Identity" | "Add" | "Sub" | "Mul" | "Div" | "Tile" | "Split" | "Expand"
                | "Pow" | "Exp" | "Log" | "Sqrt" | "Relu" | "Sigmoid" | "Tanh" => {
                    // All data-carrying inputs and outputs share the same type
                    // For Pad, only the first input (data) determines the output type, not pads or constant_value
                    let relevant_inputs: Vec<String> = if op == "Pad" {
                        node.input
                            .iter()
                            .take(1)
                            .map(|s| sanitize_name(s))
                            .collect()
                    } else {
                        node.input.iter().map(|s| sanitize_name(s)).collect()
                    };
                    let mut t_to_prop = None;

                    // Priority 1: any known i64 (except metadata inputs)
                    for (i, inp) in relevant_inputs.iter().enumerate() {
                        if inp.is_empty() {
                            continue;
                        }
                        // Skip metadata inputs
                        if op == "Reshape" && i == 1 {
                            continue;
                        }
                        if op == "Expand" && i == 1 {
                            continue;
                        }
                        if op == "Unsqueeze" && i == 1 {
                            continue;
                        }
                        if op == "Squeeze" && i == 1 {
                            continue;
                        }
                        if op == "Slice" && i >= 1 {
                            continue;
                        }
                        if op == "Tile" && i == 1 {
                            continue;
                        }
                        if op == "Split" && i == 1 {
                            continue;
                        }
                        // For Pad, pads (input 1) and constant_value (input 2) don't affect output type
                        if op == "Pad" && i >= 1 {
                            continue;
                        }
                        if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div")
                            && node.input.len() > i
                        {
                            // normally all inputs are data
                        }

                        if let Some(t) = types.get(inp) {
                            if t == "i64" {
                                t_to_prop = Some("i64".to_string());
                                break;
                            }
                            if t_to_prop.is_none() {
                                t_to_prop = Some(t.clone());
                            }
                        }
                    }
                    if t_to_prop.is_none() {
                        for out in &node.output {
                            if out.is_empty() {
                                continue;
                            }
                            if let Some(t) = types.get(&sanitize_name(out)) {
                                if t == "i64" {
                                    t_to_prop = Some("i64".to_string());
                                    break;
                                }
                                if t_to_prop.is_none() {
                                    t_to_prop = Some(t.clone());
                                }
                            }
                        }
                    }

                    if let Some(t) = t_to_prop {
                        for (i, inp) in node.input.iter().enumerate() {
                            if inp.is_empty() {
                                continue;
                            }
                            if op == "Reshape" && i == 1 {
                                continue;
                            }
                            if op == "Expand" && i == 1 {
                                continue;
                            }
                            if op == "Unsqueeze" && i == 1 {
                                continue;
                            }
                            if op == "Squeeze" && i == 1 {
                                continue;
                            }
                            if op == "Slice" && i >= 1 {
                                continue;
                            }
                            if op == "Tile" && i == 1 {
                                continue;
                            }
                            if op == "Split" && i == 1 {
                                continue;
                            }
                            if op == "Pad" && i >= 1 {
                                continue;
                            }

                            let name = sanitize_name(inp);
                            if types.get(&name) != Some(&t) {
                                types.insert(name, t.clone());
                                changed = true;
                            }
                        }
                        for out in &node.output {
                            if out.is_empty() {
                                continue;
                            }
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&t) {
                                types.insert(name, t.clone());
                                changed = true;
                            }
                        }
                    }

                    // Metadata inputs are ALWAYS i64
                    for (i, inp) in node.input.iter().enumerate() {
                        if inp.is_empty() {
                            continue;
                        }
                        let is_metadata = match op {
                            "Reshape" | "Expand" | "Unsqueeze" | "Squeeze" | "Tile" | "Split" => {
                                i == 1
                            }
                            "Slice" => i >= 1,
                            "Pad" => i == 1,
                            _ => false,
                        };
                        if is_metadata {
                            let name = sanitize_name(inp);
                            if types.get(&name) != Some(&"i64".to_string()) {
                                types.insert(name, "i64".to_string());
                                changed = true;
                            }
                        }
                    }
                }
                "Concat" => {
                    // Concat can have mixed types (e.g., embedding_concat with i64 shape + f32 weight)
                    // Determine output type from inputs, prefer f32 over i64
                    // Do NOT propagate type backwards to inputs (they may legitimately be different types)
                    let mut has_f32 = false;
                    let mut has_i64 = false;

                    // Special-case ConstantOfShape + Concat pattern: output should be f32 when ConstantOfShape value is float
                    let mut has_const_of_shape_f32 = false;
                    for inp in &node.input {
                        if inp.is_empty() {
                            continue;
                        }
                        if let Some(prod) = producer_map.get(inp) {
                            if prod.op_type == "ConstantOfShape" {
                                if constant_of_shape_output_type(prod) == "f32" {
                                    has_const_of_shape_f32 = true;
                                }
                            }
                        }
                    }

                    for inp in &node.input {
                        if inp.is_empty() {
                            continue;
                        }
                        let name = sanitize_name(inp);
                        if let Some(t) = types.get(&name) {
                            if t == "f32" {
                                has_f32 = true;
                            } else if t == "i64" {
                                has_i64 = true;
                            }
                        }
                    }

                    // Output type: f32 if any input is f32, otherwise i64 if any input is i64
                    let output_type = if has_const_of_shape_f32 {
                        Some("f32".to_string())
                    } else if has_f32 {
                        Some("f32".to_string())
                    } else if has_i64 {
                        Some("i64".to_string())
                    } else {
                        None
                    };

                    // If this is the ConstantOfShape + Concat pattern, ensure all inputs are treated as f32
                    if has_const_of_shape_f32 {
                        for inp in &node.input {
                            if inp.is_empty() {
                                continue;
                            }
                            let name = sanitize_name(inp);
                            if types.get(&name) != Some(&"f32".to_string()) {
                                types.insert(name, "f32".to_string());
                                changed = true;
                            }
                        }
                    }

                    // Only propagate to output, NOT back to inputs
                    if let Some(t) = output_type {
                        for out in &node.output {
                            if out.is_empty() {
                                continue;
                            }
                            let name = sanitize_name(out);
                            if types.get(&name) != Some(&t) {
                                types.insert(name, t.clone());
                                changed = true;
                            }
                        }
                    }
                }
                "Gather" | "GatherND" => {
                    // Input 0 and output share type. Input 1 (indices) is i64.
                    if !node.input.is_empty() && !node.input[0].is_empty() {
                        let mut t_to_prop = types.get(&sanitize_name(&node.input[0])).cloned();
                        if t_to_prop.is_none() && !node.output.is_empty() {
                            t_to_prop = types.get(&sanitize_name(&node.output[0])).cloned();
                        }
                        if let Some(t) = t_to_prop {
                            if types.get(&sanitize_name(&node.input[0])) != Some(&t) {
                                types.insert(sanitize_name(&node.input[0]), t.clone());
                                changed = true;
                            }
                            if !node.output.is_empty() {
                                if types.get(&sanitize_name(&node.output[0])) != Some(&t) {
                                    types.insert(sanitize_name(&node.output[0]), t.clone());
                                    changed = true;
                                }
                            }
                        }
                    }
                    if node.input.len() >= 2 && !node.input[1].is_empty() {
                        let name = sanitize_name(&node.input[1]);
                        if types.get(&name) != Some(&"i64".to_string()) {
                            types.insert(name, "i64".to_string());
                            changed = true;
                        }
                    }
                }
                "Where" => {
                    // Inputs 1, 2 and output share type. Input 0 is i64.
                    let mut t_to_prop = None;
                    for idx in [1, 2] {
                        if node.input.len() > idx && !node.input[idx].is_empty() {
                            if let Some(t) = types.get(&sanitize_name(&node.input[idx])) {
                                if t == "i64" {
                                    t_to_prop = Some("i64".to_string());
                                    break;
                                }
                                if t_to_prop.is_none() {
                                    t_to_prop = Some(t.clone());
                                }
                            }
                        }
                    }
                    if t_to_prop.is_none() && !node.output.is_empty() {
                        if let Some(t) = types.get(&sanitize_name(&node.output[0])) {
                            if t == "i64" {
                                t_to_prop = Some("i64".to_string());
                                break;
                            }
                            if t_to_prop.is_none() {
                                t_to_prop = Some(t.clone());
                            }
                        }
                    }

                    if let Some(t) = t_to_prop {
                        for idx in [1, 2] {
                            if node.input.len() > idx && !node.input[idx].is_empty() {
                                let name = sanitize_name(&node.input[idx]);
                                if types.get(&name) != Some(&t) {
                                    types.insert(name, t.clone());
                                    changed = true;
                                }
                            }
                        }
                        if !node.output.is_empty() && !node.output[0].is_empty() {
                            let name = sanitize_name(&node.output[0]);
                            if types.get(&name) != Some(&t) {
                                types.insert(name, t.clone());
                                changed = true;
                            }
                        }
                    }
                    if !node.input.is_empty() && !node.input[0].is_empty() {
                        let name = sanitize_name(&node.input[0]);
                        if types.get(&name) != Some(&"i64".to_string()) {
                            types.insert(name, "i64".to_string());
                            changed = true;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Default everything else to f32
    for node in nodes {
        for out in &node.output {
            if !out.is_empty() {
                let name = sanitize_name(out);
                if !types.contains_key(&name) {
                    types.insert(name, "f32".to_string());
                }
            }
        }
        for inp in &node.input {
            if !inp.is_empty() {
                let name = sanitize_name(inp);
                if !types.contains_key(&name) {
                    types.insert(name, "f32".to_string());
                }
            }
        }
    }
    types
}

pub(crate) fn generate_partitioned_graph<W: Write>(
    nodes: &[&NodeProto],           // Top level nodes
    chunk_writer: &mut Vec<String>, // Function definitions
    body_writer: &mut W,            // Forward body calls
    indent: usize,
    known_weights: &HashMap<String, (usize, usize, Vec<usize>, i32)>,
    int64_map: &HashMap<String, (Vec<i64>, Vec<usize>)>,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
    graph_inputs: &[ValueInfoProto],
    graph_outputs: &[ValueInfoProto],
) -> std::io::Result<HashMap<String, String>> {
    let chunk_size = 500;
    let total_nodes = nodes.len();
    let var_types = infer_variable_types(nodes, int64_map, graph_inputs, known_weights);
    let mut i = 0;
    let mut chunk_idx = 0;
    let mut available_vars: HashMap<String, String> = HashMap::new();
    // Initial variables in forward() scope
    for inp in graph_inputs {
        let name = sanitize_name(&inp.name);
        available_vars.insert(name.clone(), name);
    }
    while i < total_nodes {
        let end = std::cmp::min(i + chunk_size, total_nodes);
        let chunk_nodes = &nodes[i..end];
        let nodes_count = chunk_nodes.len();
        // 1. Identify Inputs: Used in chunk, defined BEFORE chunk (or global)
        let mut chunk_inputs_set = HashSet::new();
        let mut defined_raw = HashSet::new();
        let mut used_raw = HashSet::new();
        collect_recursive_metrics(chunk_nodes, &mut defined_raw, &mut used_raw);
        let mut defined_sanitized_chunk = HashSet::new();
        for d in &defined_raw {
            defined_sanitized_chunk.insert(sanitize_name(d));
        }
        for inp in &used_raw {
            let name = sanitize_name(inp);
            if name.is_empty() {
                continue;
            }
            if known_weights.contains_key(&name) {
                continue;
            }
            if int64_map.contains_key(inp) {
                continue;
            }
            if !defined_sanitized_chunk.contains(&name) {
                chunk_inputs_set.insert(name);
            }
        }
        let mut chunk_inputs: Vec<String> = chunk_inputs_set.into_iter().collect();
        chunk_inputs.sort(); // Deterministic order
        // 2. Identify Outputs: Live AFTER chunk
        let chunk_end_id = *current_id + nodes_count;
        let global_outputs: HashSet<String> =
            graph_outputs.iter().map(|o| o.name.clone()).collect();
        let _chunk_start_id = *current_id;
        let mut live_sanitized = HashSet::new();
        // Collect all potential live variables (raw names) from recursive usage
        let mut potent_live = HashSet::new();
        for d in &defined_raw {
            potent_live.insert(d.clone());
        }
        for u in &used_raw {
            potent_live.insert(u.clone());
        }
        if let Some(an) = analysis {
            for raw in potent_live {
                let sanitized = sanitize_name(&raw);
                // Filter weights
                if known_weights.contains_key(&sanitized) {
                    continue;
                }
                if int64_map.contains_key(&raw) {
                    continue;
                }
                // If defined in this chunk OR is an input arg
                let is_defined_here = chunk_nodes.iter().any(|n| n.output.contains(&raw));
                let is_input_arg = chunk_inputs.contains(&sanitized); // approximate check
                if is_defined_here || is_input_arg {
                    let last_use = *an.death_time.get(&raw).unwrap_or(&0);
                    let is_graph_out = global_outputs.contains(&raw);
                    if is_graph_out || last_use >= chunk_end_id {
                        live_sanitized.insert(sanitized);
                    }
                }
            }
        }
        let mut chunk_outputs: Vec<String> = live_sanitized.into_iter().collect();
        chunk_outputs.sort();
        // 3. Generate Function Code
        let func_name = format!("run_chunk_{}", chunk_idx);
        let mut f = Vec::new();
        let args_sig: Vec<String> = chunk_inputs
            .iter()
            .map(|n| {
                let ty = var_types.get(n).map(|s| s.as_str()).unwrap_or("f32");
                format!("{}: TensorView<'w, {}>", n, ty)
            })
            .collect();
        let ret_sig: Vec<String> = chunk_outputs
            .iter()
            .map(|n| {
                let ty = var_types.get(n).map(|s| s.as_str()).unwrap_or("f32");
                format!("TensorView<'static, {}>", ty)
            })
            .collect();
        let ret_type = if ret_sig.len() == 1 {
            ret_sig[0].clone()
        } else {
            format!("({})", ret_sig.join(", "))
        };
        writeln!(&mut f, "    #[inline(never)]")?; // Prevent inlining large chunks
        writeln!(
            &mut f,
            "    fn {}<'w>(&self, ws: &'w mut {}Workspace, {}) -> {} {{",
            func_name,
            compiler.model_name,
            args_sig.join(", "),
            ret_type
        )?;
        // Generate body
        generate_nodes(
            chunk_nodes,
            &mut f,
            2,
            known_weights,
            int64_map,
            allocator,
            analysis,
            current_id,
            compiler,
            &var_types,
        )?;
        // Return
        let ret_vals: Vec<String> = chunk_outputs
            .iter()
            .map(|s| format!("{}.to_owned()", s))
            .collect();
        if ret_vals.is_empty() {
            writeln!(&mut f, "        // No outputs")?;
        } else if ret_vals.len() == 1 {
            writeln!(&mut f, "        {}", ret_vals[0])?;
        } else {
            writeln!(&mut f, "        ({})", ret_vals.join(", "))?;
        }
        writeln!(&mut f, "    }}\n")?;
        chunk_writer.push(String::from_utf8(f).unwrap());
        // 4. Generate Call in Forward
        let tab = "    ".repeat(indent);
        let call_args: Vec<String> = chunk_inputs
            .iter()
            .map(|n| {
                if let Some(mapped) = available_vars.get(n) {
                    mapped.clone()
                } else {
                    "lele::tensor::TensorView::empty()".to_string()
                }
            })
            .collect();
        if chunk_outputs.is_empty() {
            writeln!(
                body_writer,
                "{}self.{}(ws, {});",
                tab,
                func_name,
                call_args.join(", ")
            )?;
        } else {
            let out_vars = chunk_outputs.join(", ");
            writeln!(
                body_writer,
                "{}let ({}) = self.{}(ws, {});",
                tab,
                out_vars,
                func_name,
                call_args.join(", ")
            )?;
        }
        // Update variables available in forward
        for out in &chunk_outputs {
            available_vars.insert(out.clone(), out.clone());
        }
        i += nodes_count;
        chunk_idx += 1;
    }
    Ok(var_types)
}

pub(crate) fn generate_nodes(
    nodes: &[&NodeProto],
    w: &mut dyn Write,
    indent: usize,
    known_weights: &HashMap<String, (usize, usize, Vec<usize>, i32)>,
    int64_map: &HashMap<String, (Vec<i64>, Vec<usize>)>,
    allocator: Option<&Allocator>,
    analysis: Option<&AnalysisData>,
    current_id: &mut usize,
    compiler: &Compiler,
    var_types: &HashMap<String, String>,
) -> std::io::Result<()> {
    let tab = "    ".repeat(indent);
    let mut idx = 0;
    while idx < nodes.len() {
        // Try to match patterns first
        let mut matched_len = 0;
        let remaining_nodes = &nodes[idx..];
        for pattern in &compiler.patterns {
            if let Some(len) = (pattern.matcher)(remaining_nodes) {
                (pattern.generator)(remaining_nodes, known_weights, allocator, w, indent)?;
                matched_len = len;
                break;
            }
        }
        if matched_len > 0 {
            for _ in 0..matched_len {
                *current_id += 1;
            }
            idx += matched_len;
            continue;
        }
        let node = nodes[idx];
        idx += 1;
        *current_id += 1;
        let op = node.op_type.as_str();
        let outputs: Vec<String> = node.output.iter().map(|s| sanitize_name(s)).collect();
        // Skip Code Generation for Constants that are already in weights
        if op == "Constant" && !outputs.is_empty() && known_weights.contains_key(&outputs[0]) {
            continue;
        }
        let inputs: Vec<String> = node
            .input
            .iter()
            .map(|s| {
                let name = sanitize_name(s);
                if let Some((offset, len, shape, data_type)) = known_weights.get(&name) {
                    let target_type = var_types.get(&name).map(|s| s.as_str()).unwrap_or("f32");
                    if target_type == "i64" {
                        match data_type {
                            7 => format!("self.weight_i64({}, {}, &{:?})", offset, len, shape),
                            6 => format!("self.weight_i32_i64({}, {}, &{:?})", offset, len, shape),
                            _ => format!("self.weight_i64({}, {}, &{:?})", offset, len, shape), // fallback
                        }
                    } else {
                        match data_type {
                            1 => format!("self.weight_f32({}, {}, &{:?})", offset, len, shape),
                            2 => format!("self.weight_u8({}, {}, &{:?})", offset, len, shape),
                            3 => format!("self.weight_i8({}, {}, &{:?})", offset, len, shape),
                            6 => format!("self.weight_i32_f32({}, {}, &{:?})", offset, len, shape),
                            7 => format!("self.weight_i64_f32({}, {}, &{:?})", offset, len, shape),
                            10 => format!("self.weight_f16({}, {}, &{:?})", offset, len, shape),
                            _ => format!("self.weight_f32({}, {}, &{:?})", offset, len, shape),
                        }
                    }
                } else {
                    name
                }
            })
            .collect();
        // Unused check
        let is_unused = if let Some(an) = analysis {
            if op == "If" || op == "Loop" {
                false
            }
            // Conservative for control flow
            else {
                node.output.iter().all(|out| {
                    if out.is_empty() {
                        return true;
                    }
                    /* Graph outputs check removed here as we don't have access to graph outputs easily in this Func,
                    but liveness analysis should handle it via infinity use */
                    an.uses.get(out).map(|u| u.is_empty()).unwrap_or(true)
                })
            }
        } else {
            false
        };
        if is_unused && !node.output.is_empty() {
            // writeln!(w, "{}// Op {} {} skipped (unused)", tab, op, outputs.join(","))?;
            continue;
        }
        let is_i64 = outputs
            .get(0)
            .and_then(|out| var_types.get(out))
            .map(|t| t == "i64")
            .unwrap_or(false);
        let buf_expr = if let Some(alloc) = allocator {
            if !node.output.is_empty() {
                if let Some(&idx) = alloc.tensor_to_buffer.get(&node.output[0]) {
                    if is_i64 {
                        format!("&mut buf_{}", outputs[0])
                    } else {
                        format!("&mut ws.buf_{}", idx)
                    }
                } else {
                    if !outputs.is_empty() {
                        format!("&mut buf_{}", outputs[0])
                    } else {
                        String::new()
                    }
                }
            } else {
                String::new()
            }
        } else {
            if !outputs.is_empty() {
                format!("&mut buf_{}", outputs[0])
            } else {
                String::new()
            }
        };

        if is_i64 && !outputs.is_empty() && buf_expr.starts_with("&mut buf_") {
            writeln!(w, "{}let mut buf_{} = Vec::<i64>::new();", tab, outputs[0])?;
        } else if !is_i64 && !outputs.is_empty() && buf_expr.starts_with("&mut buf_") {
            // Built-in handlers might not need Vec<f32> if they don't use buf_expr,
            // but we'll generate it by default if expected.
            // Some ops like Split/LSTM manage their own buffers.
            if op != "DynamicQuantizeLinear" && op != "LSTM" && op != "Split" {
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, outputs[0])?;
            }
        }
        let mut ctx = OpContext {
            node,
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            buf_expr,
            indent,
            known_weights,
            int64_map,
            allocator,
            analysis,
            current_id,
            compiler,
            var_types,
        };

        // 1. Override
        if let Some(handler) = compiler.overrides.get(op) {
            handler(node, &inputs, &outputs, &ctx.buf_expr, w, indent)?;
            continue;
        }

        // 2. Built-in
        if super::ops::dispatch_builtin(&mut ctx, w)? {
            continue;
        }

        // 3. Fallback
        eprintln!(
            "Warning: Unrecognized operator '{}' (node: {})",
            op, node.name
        );
        for (idx, out_name) in outputs.iter().enumerate() {
            writeln!(
                w,
                "{}let {} = lele::tensor::TensorView::empty(); // Unimplemented {} out {}",
                tab, out_name, op, idx
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::model::onnx_proto::NodeProto;
    use std::collections::HashMap;

    #[test]
    fn test_custom_op_override() {
        let mut compiler = Compiler::new();
        compiler = compiler.with_override("CustomOp", |_node, inputs, outputs, _buf, w, indent| {
            let tab = "    ".repeat(indent);
            writeln!(w, "{}// Custom implementation for {}", tab, outputs[0])?;
            writeln!(
                w,
                "{}let {} = custom_kernel(&{});",
                tab, outputs[0], inputs[0]
            )
        });

        let node = NodeProto {
            input: vec!["input1".to_string()],
            output: vec!["output1".to_string()],
            op_type: "CustomOp".to_string(),
            ..Default::default()
        };

        let mut output = Vec::new();
        let mut current_id = 0;
        let known_weights = HashMap::new();
        let int64_map = HashMap::new();
        let var_types = HashMap::new();

        generate_nodes(
            &[&node],
            &mut output,
            0,
            &known_weights,
            &int64_map,
            None,
            None,
            &mut current_id,
            &compiler,
            &var_types,
        )
        .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("// Custom implementation for output1"));
        assert!(result.contains("let output1 = custom_kernel(&input1);"));
    }

    #[test]
    fn test_builtin_dispatch() {
        let compiler = Compiler::new();
        let node = NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["c".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let mut output = Vec::new();
        let mut current_id = 0;
        let known_weights = HashMap::new();
        let int64_map = HashMap::new();
        let var_types = HashMap::new();

        generate_nodes(
            &[&node],
            &mut output,
            0,
            &known_weights,
            &int64_map,
            None,
            None,
            &mut current_id,
            &compiler,
            &var_types,
        )
        .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("let mut buf_c = Vec::<f32>::new();"));
        assert!(result.contains("let c = lele::kernels::add(&a, &b, &mut buf_c);"));
    }
}
