#![allow(dead_code)]
use std::{
    fs::{File, OpenOptions},
    io::{self, Seek, Write},
};

const PAGESIZE: usize = 4096;
const MAX_NUM_OF_KEYS: usize = 30;
const MAX_KEY_SIZE: usize = 4;

#[repr(u8)]
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

struct Slot {
    key: Vec<u8>,
    offset: u16,
    length: u16,
}

struct PageHeader {
    node_type: NodeType,
    unknown: u8,
}

struct InternalNode {
    header: PageHeader,
    keys: Vec<i32>,
    children: Vec<u64>,
}

struct LeafNode {
    header: PageHeader,
    free_list_offset: u16,
    slot_count: u16,
    slots: Vec<Slot>,
    data: Vec<u8>, //  data will be stored starting at the end of the page
}

impl LeafNode {
    fn serialize(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.push(u8::from(&self.header.node_type));
        bytes.extend(self.header.unknown.to_le_bytes());
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
            header: PageHeader { node_type, unknown },
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
        println!("start {}, end {}", start, end);
        println!("data length {}", self.data.len());
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
}

enum BPlusTreeNode {
    Internal(InternalNode),
    Leaf(LeafNode),
}

struct BPlusTree {
    root: BPlusTreeNode,
    file: File,
}

impl BPlusTree {
    fn new(filename: &str) -> Self {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(filename)
            .expect("Failed to open file");
        let header = PageHeader {
            node_type: NodeType::Header,
            unknown: 0,
        };
        let slots_available = MAX_NUM_OF_KEYS;
        let memory_for_slots = slots_available * std::mem::size_of::<Slot>();
        let data_size = PAGESIZE
            - std::mem::size_of::<PageHeader>()
            - std::mem::size_of::<u16>()
            - std::mem::size_of::<u16>()
            - memory_for_slots;
        let root = BPlusTreeNode::Leaf(LeafNode {
            header,
            free_list_offset: data_size as u16,
            slot_count: 0,
            slots: Vec::new(),
            data: vec![0u8; data_size],
        });
        BPlusTree { root, file }
    }

    fn insert(&mut self, key: &[u8], value: &[u8]) -> bool {
        match &mut self.root {
            BPlusTreeNode::Internal(_internal) => (),
            BPlusTreeNode::Leaf(leaf) => {
                leaf.insert(key, value);
                let leaf_bytes = leaf.serialize();
                self.file
                    .seek(io::SeekFrom::Start(0))
                    .expect("error while seeking");
                self.file
                    .write_all(&leaf_bytes)
                    .expect("error while writing");
                self.file.sync_all().expect("failed to sync data");
            }
        };
        true
    }
}

fn main() {
    let filename = "btree.db";
    let mut tree = BPlusTree::new(filename);
    tree.insert("key".as_bytes(), "value".as_bytes());
}
