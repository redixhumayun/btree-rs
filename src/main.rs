#![allow(dead_code)]
use std::{
    collections::{HashMap, VecDeque},
    fs::{File, OpenOptions},
    io::{self, Read, Seek, Write},
    sync::{Arc, Mutex},
};

const PAGESIZE: usize = 4096;
const MAX_NUM_OF_KEYS: usize = 4;
const MAX_KEY_SIZE: usize = 4;

#[repr(u8)]
#[derive(Debug, Clone)]
enum NodeType {
    Header,
    Internal,
    Leaf,
}

impl From<&NodeType> for u8 {
    fn from(value: &NodeType) -> Self {
        match value {
            NodeType::Header => 0,
            NodeType::Internal => 1,
            NodeType::Leaf => 2,
        }
    }
}

impl From<u8> for NodeType {
    fn from(value: u8) -> Self {
        match value {
            0 => NodeType::Header,
            1 => NodeType::Internal,
            2 => NodeType::Leaf,
            _ => panic!("Invalid node type"),
        }
    }
}

#[derive(Clone, Copy)]
struct PageNumber(u8);

impl PageNumber {
    fn get(&self) -> u8 {
        self.0
    }
}

impl From<u8> for PageNumber {
    fn from(value: u8) -> Self {
        PageNumber(value)
    }
}

impl From<PageNumber> for u8 {
    fn from(page: PageNumber) -> Self {
        page.0
    }
}

impl From<&PageNumber> for u8 {
    fn from(page: &PageNumber) -> Self {
        page.0
    }
}

#[derive(Debug, Clone)]
struct PageHeader {
    node_type: NodeType,
    page_number: u8,
}

impl PageHeader {
    fn serialize(&self, buf: &mut Vec<u8>) {
        buf.push(u8::from(&self.node_type));
        buf.extend(self.page_number.to_le_bytes());
    }

    fn deserialize(buf: Vec<u8>) -> Self {
        let node_type = NodeType::from(buf[0]);
        let page_number = buf[1];
        PageHeader {
            node_type,
            page_number,
        }
    }
}

#[derive(Debug, Clone)]
struct InternalNode {
    header: PageHeader,
    num_of_children: u8,
    keys: Vec<i32>,
    children: Vec<u8>,
    right_child: Option<u8>,
}

impl InternalNode {
    fn new(page_number: u8) -> Self {
        InternalNode {
            header: PageHeader {
                node_type: NodeType::Internal,
                page_number,
            },
            num_of_children: 0,
            keys: Vec::new(),
            children: Vec::new(),
            right_child: None,
        }
    }

    fn get_page_number(&self) -> u8 {
        self.header.page_number
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        self.header.serialize(&mut bytes);
        for key in &self.keys {
            bytes.extend(key.to_le_bytes());
        }
        for child in &self.children {
            bytes.extend(child.to_le_bytes());
        }
        if let Some(right_child) = self.right_child {
            bytes.push(right_child);
        }
        bytes
    }

    fn deserialize(bytes: &[u8]) -> io::Result<Self> {
        let page_header = PageHeader::deserialize(bytes.to_vec());
        let num_of_children = bytes[2];
        let mut offset = 3;
        let mut keys = Vec::new();
        let mut children = Vec::new();
        for _ in 0..num_of_children {
            let key = i32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            keys.push(key);
            offset += MAX_KEY_SIZE;
        }
        for _ in 0..num_of_children {
            let child = bytes[offset];
            children.push(child);
            offset += 1;
        }
        let right_child = if offset < bytes.len() {
            Some(bytes[offset])
        } else {
            None
        };
        Ok(InternalNode {
            header: page_header,
            right_child,
            num_of_children,
            keys,
            children,
        })
    }

    fn insert(&mut self, key: i32, left_child: u8) -> bool {
        let key_insert_position = self.keys.binary_search(&key).unwrap_or_else(|index| index);
        self.keys.insert(key_insert_position, key);
        self.children.insert(key_insert_position, left_child);
        true
    }

    fn set_child(&mut self, index: usize, left_child: u8) -> bool {
        if index >= self.keys.len() {
            panic!("index is beyond keys length");
        }
        self.children[index] = left_child;
        true
    }

    fn insert_for_promoted_key(&mut self, key: i32, left_child: u8, right_child: u8) -> bool {
        let key_insert_position = self.keys.binary_search(&key).unwrap_or_else(|index| index);
        if key_insert_position == self.keys.len() {
            self.right_child = Some(right_child);
        } else {
            self.children[key_insert_position + 1] = right_child;
        }
        self.keys.insert(key_insert_position, key);
        self.children.insert(key_insert_position, left_child);
        true
    }

    fn delete(&mut self, key: i32) -> bool {
        //  TODO: Check if this becomes less than half full and force a merge
        let key_index = self.keys.binary_search(&key).unwrap();
        self.keys.remove(key_index);
        self.children.remove(key_index);
        true
    }

    fn search(&self, key: i32) -> PageNumber {
        match self.keys.binary_search(&key) {
            Ok(key_index) => {
                if key_index == self.keys.len() - 1 {
                    if let Some(right_child) = self.right_child {
                        return PageNumber(right_child);
                    }
                    panic!("could not find the right child of an internal node");
                }
                return PageNumber(self.children[key_index]);
            }
            Err(insertion_index) => {
                if insertion_index == self.keys.len() {
                    if let Some(right_child) = self.right_child {
                        return PageNumber(right_child);
                    }
                    panic!("could not find right child of an internal node");
                }
                return PageNumber(self.children[insertion_index]);
            }
        }
    }

    fn split(&mut self, new_page_number: u8) -> (i32, Self, InternalNode) {
        let median_index = self.keys.len() / 2;
        let key_to_promote = self.keys[median_index];

        //  create the new internal node and transfer keys and values past midpoint and the right child over
        let mut new_internal_node = InternalNode::new(new_page_number);
        new_internal_node.keys = self.keys.drain(median_index + 1..).collect();
        new_internal_node.children = self.children.drain(median_index + 1..).collect();
        new_internal_node.right_child = self.right_child;

        //  left child of promoted key becomes right child of original node
        self.right_child = Some(self.children[median_index]);

        //  remove the promoted key and it's child from the original node
        self.keys.remove(median_index);
        self.children.remove(median_index);

        (key_to_promote, self.clone(), new_internal_node)
    }
}

#[derive(Debug, Clone)]
struct Slot {
    key: Vec<u8>,
    offset: u16,
    length: u16,
}

#[derive(Debug, Clone)]
struct LeafNode {
    header: PageHeader,
    free_list_offset: u16,
    slot_count: u16,
    slots: Vec<Slot>,
    data: Vec<u8>, //  data will be stored starting at the end of the page
}

impl LeafNode {
    fn new(page_number: u8) -> Self {
        let slots_available = MAX_NUM_OF_KEYS;
        let memory_for_slots = slots_available * std::mem::size_of::<Slot>();
        let data_size = PAGESIZE
            - std::mem::size_of::<PageHeader>()
            - std::mem::size_of::<u16>()    //  free list offset
            - std::mem::size_of::<u16>()    //  slot count
            - memory_for_slots;
        LeafNode {
            header: PageHeader {
                node_type: NodeType::Leaf,
                page_number,
            },
            free_list_offset: data_size as u16,
            slot_count: 0,
            slots: Vec::new(),
            data: vec![0u8; data_size],
        }
    }

    fn get_page_number(&self) -> u8 {
        self.header.page_number
    }

    fn serialize(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.push(u8::from(&self.header.node_type));
        bytes.extend(self.header.page_number.to_le_bytes());
        bytes.extend(self.free_list_offset.to_le_bytes());
        bytes.extend_from_slice(&self.slot_count.to_le_bytes());
        for slot in &self.slots {
            bytes.extend_from_slice(&slot.key);
            bytes.extend_from_slice(&slot.offset.to_le_bytes());
            bytes.extend_from_slice(&slot.length.to_le_bytes());
        }
        bytes.extend(&self.data);
        bytes
    }

    fn deserialize(bytes: &[u8]) -> io::Result<Self> {
        let node_type = NodeType::from(bytes[0]);
        let unknown = bytes[1];
        let free_list_offset = u16::from_le_bytes([bytes[2], bytes[3]]);
        let slot_count = u16::from_le_bytes([bytes[4], bytes[5]]);

        let mut offset = 6;
        let mut slots = Vec::new();
        for _ in 0..slot_count {
            let key_end = offset + MAX_KEY_SIZE;
            let key = bytes[offset..key_end].to_vec();
            offset = key_end;

            let slot_offset = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;

            let length = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;

            slots.push(Slot {
                key,
                offset: slot_offset,
                length,
            });
        }

        let data = bytes[offset..].to_vec();

        Ok(LeafNode {
            header: PageHeader {
                node_type,
                page_number: unknown,
            },
            free_list_offset,
            slot_count,
            slots,
            data,
        })
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> bool {
        if key.len() > MAX_KEY_SIZE {
            return false;
        }
        let value_len = value.len() as u16;

        //  check if the page needs to be split
        if self.slots.len() == MAX_NUM_OF_KEYS {
            return false;
        }

        self.free_list_offset -= value_len;
        let start = self.free_list_offset as usize;
        let end = start + value_len as usize;
        self.data[start..end].copy_from_slice(&value);

        let slot = Slot {
            key: key.to_vec(),
            offset: self.free_list_offset,
            length: value_len,
        };
        let position = self
            .slots
            .binary_search_by_key(&key, |s| &s.key)
            .unwrap_or_else(|existing_index| existing_index);
        self.slots.insert(position, slot);
        self.slot_count += 1;
        true
    }

    /// This method will take a key and remove it from the leaf. It does this by finding the slot to remove
    /// It shifts all the data in the preceding slots over by the size of the value being removed and then zeroes out the shifted values
    fn delete(&mut self, key: &[u8]) -> bool {
        if key.len() > MAX_KEY_SIZE {
            return false;
        }
        let position = self.slots.binary_search_by_key(&key, |s| &s.key).unwrap();
        let slot_found = self.slots.get(position).unwrap();
        assert!(
            slot_found.key == key,
            "the key found was not the key being searched"
        );

        let tombstone: Vec<u8> = vec![0xDE, 0xAD];

        let start = slot_found.offset as usize;
        let end = start + slot_found.length as usize;
        if end - start >= tombstone.len() {
            self.data[start..start + tombstone.len()].copy_from_slice(&tombstone);
            self.data[start + tombstone.len()..end].fill(0);
        } else {
            panic!("Could not insert a tombstone because the size of the value is smaller than the tombstone");
        }
        self.slots.remove(position);
        true
    }

    fn search(&self, key: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let slot_index = self.slots.binary_search_by_key(&key, |s| &s.key).unwrap();
        let offset = self.slots[slot_index].offset;
        let length = self.slots[slot_index].length;
        let start = offset as usize;
        let end = start + length as usize;
        let value = &self.data[start..end];
        Ok(value.to_vec())
    }

    fn split(&mut self, new_page_number: u8) -> (Vec<u8>, Self) {
        let midpoint = self.slots.len() / 2;

        let key_to_promote = self.slots[midpoint].key.clone();
        let keys_to_move: Vec<Slot> = self.slots[midpoint..]
            .iter()
            .map(|slot| slot.clone())
            .collect();
        let key_value_pairs: Vec<_> = keys_to_move
            .iter()
            .map(|slot| {
                let start = slot.offset as usize;
                let end = start + slot.length as usize;
                let mut data = vec![0; end - start];
                data.copy_from_slice(&self.data[start..end]);
                (slot.key.clone(), data)
            })
            .collect();
        for slot in keys_to_move {
            self.delete(&slot.key);
        }
        let mut new_leaf = LeafNode::new(new_page_number);
        for key_value in key_value_pairs {
            new_leaf.insert(&key_value.0, &key_value.1);
        }
        (key_to_promote, new_leaf)
    }
}

#[derive(Clone)]
enum BPlusTreeNode {
    Internal(InternalNode),
    Leaf(LeafNode),
}

impl std::fmt::Debug for BPlusTreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BPlusTreeNode::Internal(node) => {
                write!(
                    f,
                    "InternalNode(Page: {})\n  Keys: {:?}\n  Children: {:?}\n  Right Child(Page: {:?})\n",
                    node.header.page_number, node.keys, node.children, node.right_child
                )
            }
            BPlusTreeNode::Leaf(node) => {
                write!(
                    f,
                    "LeafNode(Page: {})\n  Keys: {:?}\n",
                    node.header.page_number,
                    node.slots
                        .iter()
                        .map(|slot| String::from_utf8(slot.key.clone())
                            .unwrap_or_else(|_| "<Invalid UTF-8>".to_string()))
                        .collect::<Vec<_>>()
                )
            }
        }
    }
}

struct BPlusTree {
    root: BPlusTreeNode,
    file: File,
    next_page_number: u8,
    cache: Arc<Mutex<HashMap<u8, BPlusTreeNode>>>,
}

impl std::fmt::Debug for BPlusTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache = self.cache.lock().unwrap();
        let mut nodes = vec![(&self.root, 0)];

        while let Some((node, level)) = nodes.pop() {
            match node {
                BPlusTreeNode::Internal(internal) => {
                    writeln!(
                        f,
                        "{}InternalNode(Page: {})",
                        "  ".repeat(level),
                        internal.header.page_number
                    )?;
                    writeln!(f, "{}Keys: {:?}", "  ".repeat(level + 1), internal.keys)?;
                    writeln!(
                        f,
                        "{}Children: {:?}",
                        "  ".repeat(level + 1),
                        internal.children
                    )?;

                    for &child_page in internal.children.iter().rev() {
                        if let Some(child_node) = cache.get(&child_page) {
                            nodes.push((child_node, level + 1));
                        }
                    }
                    if let Some(right_child) = internal.right_child {
                        if let Some(child_node) = cache.get(&right_child) {
                            nodes.insert(0, (child_node, level + 1));
                        }
                    }
                }
                BPlusTreeNode::Leaf(leaf) => {
                    writeln!(
                        f,
                        "{}LeafNode(Page: {})",
                        "  ".repeat(level),
                        leaf.header.page_number
                    )?;
                    writeln!(
                        f,
                        "{}Keys: {:?}",
                        "  ".repeat(level + 1),
                        leaf.slots
                            .iter()
                            .map(|slot| String::from_utf8(slot.key.clone())
                                .unwrap_or_else(|_| "<Invalid UTF-8>".to_string()))
                            .collect::<Vec<_>>()
                    )?;
                }
            }
        }

        Ok(())
    }
}

impl BPlusTree {
    fn new(filename: &str) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(filename)
            .expect("Failed to open file");
        let root = BPlusTreeNode::Leaf(LeafNode::new(0));
        let cache = Arc::new(Mutex::new(HashMap::new()));
        cache.lock().unwrap().insert(0, root.clone());
        BPlusTree {
            root,
            file,
            next_page_number: 1,
            cache,
        }
    }

    fn tree_for_printing(
        file: &str,
        root: BPlusTreeNode,
        cache: Arc<Mutex<HashMap<u8, BPlusTreeNode>>>,
    ) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .open(file)
            .expect("failed to open file");
        BPlusTree {
            root,
            file,
            next_page_number: 0,
            cache,
        }
    }

    fn get_root_page_number(&self) -> PageNumber {
        match &self.root {
            BPlusTreeNode::Internal(internal) => PageNumber(internal.header.page_number),
            BPlusTreeNode::Leaf(leaf) => PageNumber(leaf.header.page_number),
        }
    }

    fn get_new_page_number(&mut self) -> u8 {
        let ret = self.next_page_number;
        self.next_page_number += 1;
        ret
    }

    fn page_num_to_offset(page_number: u8) -> u64 {
        page_number as u64 * PAGESIZE as u64
    }

    fn write_to_cache(&self, page_number: u8, node: &BPlusTreeNode) {
        self.cache.lock().unwrap().insert(page_number, node.clone());
    }

    fn write_page(&mut self, node: &BPlusTreeNode, page_number: PageNumber) {
        let bytes = match node {
            BPlusTreeNode::Internal(internal) => internal.serialize(),
            BPlusTreeNode::Leaf(leaf) => leaf.serialize(),
        };
        self.file
            .seek(io::SeekFrom::Start(BPlusTree::page_num_to_offset(
                page_number.into(),
            )))
            .unwrap();
        self.file.write_all(&bytes).expect("failed to write node");
        self.file.sync_all().expect("unable to sync the file");
    }

    fn read_page(&mut self, page_number: PageNumber) -> BPlusTreeNode {
        if let Some(node) = self.cache.lock().unwrap().get(&page_number.get()) {
            return node.clone();
        }

        let offset = BPlusTree::page_num_to_offset(page_number.into());
        self.file.seek(io::SeekFrom::Start(offset)).unwrap();
        let mut buf = vec![0; PAGESIZE];
        self.file.read_exact(&mut buf).unwrap();

        let node_type = NodeType::from(buf[0]);
        match node_type {
            NodeType::Header => panic!("should not have found header page"),
            NodeType::Internal => BPlusTreeNode::Internal(InternalNode::deserialize(&buf).unwrap()),
            NodeType::Leaf => BPlusTreeNode::Leaf(LeafNode::deserialize(&buf).unwrap()),
        }
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> bool {
        let mut path: VecDeque<PageNumber> = VecDeque::new();
        let root_page_num = self.get_root_page_number();
        let mut current_root = self.read_page(root_page_num);
        loop {
            match current_root {
                BPlusTreeNode::Internal(internal) => {
                    let original_internal_page_num = internal.header.page_number;
                    let key = i32::from_le_bytes(key.try_into().unwrap());
                    path.push_back(original_internal_page_num.into());
                    let child_page_num = internal.search(key);
                    let child_node = self.read_page(child_page_num);
                    current_root = child_node.clone();
                }
                BPlusTreeNode::Leaf(mut leaf) => {
                    //  check if the leaf has space
                    let original_leaf_page_num = leaf.header.page_number;
                    if leaf.insert(key, value) {
                        self.write_to_cache(leaf.get_page_number(), &BPlusTreeNode::Leaf(leaf));
                        return true;
                    }

                    //  the leaf does not have space, need to perform a split
                    let new_leaf_node_page_num = self.get_new_page_number();
                    let (key_to_promote, mut new_leaf_node) = leaf.split(new_leaf_node_page_num);
                    if i32::from_le_bytes(key.try_into().unwrap()) <= key_to_promote[0].into() {
                        leaf.insert(key, value);
                    } else {
                        new_leaf_node.insert(key, value);
                    }

                    let original_leaf_node = leaf.clone();
                    self.write_to_cache(
                        original_leaf_page_num,
                        &BPlusTreeNode::Leaf(original_leaf_node.clone()),
                    );
                    self.write_to_cache(
                        new_leaf_node_page_num,
                        &BPlusTreeNode::Leaf(new_leaf_node.clone()),
                    );

                    let mut key: i32 = key_to_promote[0].into();
                    if path.is_empty() {
                        //  No parent, create new root
                        let new_root_page_number = self.get_new_page_number();
                        let mut new_root = InternalNode::new(new_root_page_number);
                        new_root.insert(key, original_leaf_page_num);
                        new_root.right_child = Some(new_leaf_node.get_page_number());
                        self.root = BPlusTreeNode::Internal(new_root);
                        self.write_to_cache(new_root_page_number, &self.root);
                        self.write_to_cache(original_leaf_page_num, &BPlusTreeNode::Leaf(leaf));
                        return true;
                    } else {
                        //  Traverse the stack back up, propogating splits as necessary
                        while let Some(parent_page_num) = path.pop_back() {
                            let parent_node = self.read_page(parent_page_num);
                            let mut parent_node = match parent_node {
                                BPlusTreeNode::Internal(node) => node,
                                BPlusTreeNode::Leaf(_) => {
                                    panic!("found a leaf node where internal node was expected")
                                }
                            };

                            //  No need to split the parent node
                            if parent_node.insert(key, original_leaf_page_num) {
                                let key_insert_pos =
                                    parent_node.keys.iter().position(|k| *k == key).unwrap();
                                if key_insert_pos == parent_node.keys.len() - 1 {
                                    parent_node.right_child = Some(new_leaf_node_page_num);
                                } else {
                                    parent_node.set_child(key_insert_pos, new_leaf_node_page_num);
                                }
                                self.write_to_cache(
                                    parent_page_num.into(),
                                    &BPlusTreeNode::Internal(parent_node),
                                );
                                self.write_to_cache(
                                    original_leaf_page_num,
                                    &BPlusTreeNode::Leaf(original_leaf_node),
                                );
                                self.write_to_cache(
                                    new_leaf_node_page_num,
                                    &BPlusTreeNode::Leaf(new_leaf_node),
                                );
                                break;
                            }

                            //  Need to split the parent node
                            let new_internal_node_page_number = self.get_new_page_number();
                            let (key_to_promote, left_internal_node, right_internal_node) =
                                parent_node.split(new_internal_node_page_number);
                            let left_page_num = left_internal_node.header.page_number;
                            let right_page_num = right_internal_node.header.page_number;
                            self.write_to_cache(
                                left_page_num,
                                &BPlusTreeNode::Internal(left_internal_node),
                            );
                            self.write_to_cache(
                                right_page_num,
                                &BPlusTreeNode::Internal(right_internal_node),
                            );
                            if path.is_empty() {
                                //  need to create a new root
                                let new_root_page_num = self.get_new_page_number();
                                let mut new_root = InternalNode::new(new_root_page_num);
                                new_root.insert(key_to_promote, left_page_num);
                                new_root.right_child = Some(right_page_num);
                                self.root = BPlusTreeNode::Internal(new_root);
                                self.write_to_cache(new_root_page_num, &self.root);
                                break;
                            }
                            key = key_to_promote;
                        }
                        return true;
                    }
                }
            };
        }
    }

    fn search(&mut self, key: &[u8]) -> Vec<u8> {
        let root_page_num = self.get_root_page_number();
        let mut current_node = self.read_page(root_page_num);
        loop {
            match current_node {
                BPlusTreeNode::Internal(internal) => {
                    let bytes: [u8; 4] = key.try_into().unwrap();
                    let key = i32::from_le_bytes(bytes);
                    let page_number = internal.search(key);
                    current_node = self.read_page(page_number);
                }
                BPlusTreeNode::Leaf(leaf) => {
                    let value = leaf.search(key).unwrap();
                    return value;
                }
            }
        }
    }
}

fn main() {
    let filename = "btree.db";
    let mut tree = BPlusTree::new(filename);
    assert!(tree.insert(&1_i32.to_le_bytes(), "value1".as_bytes()));
    assert!(tree.insert(&2_i32.to_le_bytes(), "value2".as_bytes()));
    assert!(tree.insert(&3_i32.to_le_bytes(), "value3".as_bytes()));
    assert!(tree.insert(&4_i32.to_le_bytes(), "value4".as_bytes()));
    assert!(tree.insert(&5_i32.to_le_bytes(), "value5".as_bytes()));

    let result = tree.search(&5_i32.to_le_bytes());
    assert_eq!(result, "value5".as_bytes());
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use crate::BPlusTree;

    fn print_tree(tree: &mut BPlusTree, filename: &str) {
        let root = tree.read_page(tree.get_root_page_number());
        let tree_for_printing =
            BPlusTree::tree_for_printing(filename, root, Arc::clone(&tree.cache));
        println!("the tree {:?}", tree_for_printing);
    }

    #[test]
    fn check_leaf_split_sequential_writes() {
        let filename = "btree-test.db";
        let mut tree = BPlusTree::new(filename);

        tree.insert(&1_i32.to_le_bytes(), "value1".as_bytes());
        tree.insert(&2_i32.to_le_bytes(), "value2".as_bytes());
        tree.insert(&3_i32.to_le_bytes(), "value3".as_bytes());
        tree.insert(&4_i32.to_le_bytes(), "value4".as_bytes());
        tree.insert(&5_i32.to_le_bytes(), "value5".as_bytes());

        print_tree(&mut tree, filename);

        let result = tree.search(&3_i32.to_le_bytes());
        assert_eq!(result, "value3".as_bytes());

        tree.insert(&6_i32.to_le_bytes(), "value6".as_bytes());
        tree.insert(&7_i32.to_le_bytes(), "value7".as_bytes());

        print_tree(&mut tree, filename);

        let result = tree.search(&5_i32.to_le_bytes());
        assert_eq!(result, "value5".as_bytes());

        std::fs::remove_file(PathBuf::from(filename)).unwrap();
    }

    #[test]
    fn check_leaf_split_random_writes() {
        let filename = "btree-test.db";
        let mut tree = BPlusTree::new(filename);

        tree.insert(&1_i32.to_le_bytes(), "value1".as_bytes());
        tree.insert(&4_i32.to_le_bytes(), "value4".as_bytes());
        tree.insert(&8_i32.to_le_bytes(), "value8".as_bytes());
        tree.insert(&9_i32.to_le_bytes(), "value9".as_bytes());
        tree.insert(&15_i32.to_le_bytes(), "value15".as_bytes());
        tree.insert(&7_i32.to_le_bytes(), "value7".as_bytes());
        tree.insert(&6_i32.to_le_bytes(), "value6".as_bytes());
        tree.insert(&5_i32.to_le_bytes(), "value5".as_bytes());
        print_tree(&mut tree, filename);
        assert_eq!(tree.search(&9_i32.to_le_bytes()), "value9".as_bytes());
    }
}
