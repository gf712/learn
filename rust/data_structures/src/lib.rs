mod data_structures {
    #[derive(Debug)]
    struct Node<'a, T> {
        data: &'a T,
        next_node: Option<Box<Node<'a, T>>>,
    }

    pub struct LinkedList<'a, T> {
        front: Option<Box<Node<'a, T>>>,
    }

    impl<'a, T: std::fmt::Debug> LinkedList<'a, T> {
        pub fn new() -> LinkedList<'a, T> {
            LinkedList::<T> { front: None }
        }

        pub fn front(&self) -> Option<&'a T> {
            if !self.empty() {
                Some(self.front.as_ref().unwrap().data)
            } else {
                None
            }
        }

        pub fn back(&self) -> Option<&'a T> {
            if self.empty() {
                return None;
            }
            let mut node = self.front.as_ref().unwrap();
            loop {
                match &node.next_node {
                    Some(x) => node = &x,
                    None => return Some(node.data),
                }
            }
        }

        pub fn empty(&self) -> bool {
            match self.front {
                Some(_) => false,
                None => true,
            }
        }

        pub fn size(&self) -> usize {
            match &self.front {
                Some(x) => 1 + LinkedList::<'a, T>::size_helper(x),
                None => 0,
            }
        }

        pub fn push_back(&mut self, data: &'a T) {
            match &self.front {
                Some(x) => LinkedList::<'a, T>::push_back_helper(self.front.as_mut().unwrap(), &data),
                None => self.front = Some(Box::new(Node::<'a, T> {
                    data,
                    next_node: None,
                }))
            }
        }

        pub fn pop_back(&mut self) {
            match &self.front {
                Some(x) => {
                    if LinkedList::<'a, T>::has_next(x) {
                        LinkedList::<'a, T>::pop_back_helper(self.front.as_mut().unwrap());
                    }
                    else {
                        self.front = None;
                    }
                },
                _ => ()
            }
        }

        fn pop_back_helper(node: &mut Node<'a, T>) {
            match &mut node.next_node {
                Some(x) => {
                    if LinkedList::<'a, T>::has_next(x) {
                        LinkedList::<'a, T>::pop_back_helper(x)
                    }
                    else {
                        node.next_node = None;
                    }
                },
                _ => (),
            }
        }

        fn has_next(node: &Node<'a, T>) -> bool {
            match node.next_node {
                Some(_) => true,
                None => false,
            }
        }

        fn push_back_helper(node: &mut Node<'a, T>, data: &'a T) {
            match &mut node.next_node {
                Some(x) => LinkedList::<'a, T>::push_back_helper(x, &data),
                None => {
                    node.next_node = Some(Box::new(Node::<'a, T> {
                        data,
                        next_node: None,
                    }))
                }
            }
        }

        fn size_helper(node: &Node<'a, T>) -> usize {
            match &node.next_node {
                Some(x) => 1 + LinkedList::<'a, T>::size_helper(x),
                None => 0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::data_structures::LinkedList;

    #[test]
    fn empty_linked_list() {
        let list = LinkedList::<f32>::new();
        assert_eq!(list.empty(), true);
        assert_eq!(list.front(), None);
    }

    #[test]
    fn push_back_linked_list() {
        let value: f32 = 1.0;
        let mut list = LinkedList::<f32>::new();
        list.push_back(&value);
        assert_eq!(*list.back().unwrap(), value);
        list.push_back(&value);
        assert_eq!(*list.back().unwrap(), value);
        assert_eq!(list.empty(), false);
    }


    #[test]
    fn pop_back_linked_list() {
        let value1: f32 = 1.0;
        let value2: f32 = 2.0;
        let mut list = LinkedList::<f32>::new();
        list.push_back(&value1);
        list.push_back(&value2);
        assert_eq!(*list.back().unwrap(), value2);
        list.pop_back();
        assert_eq!(*list.back().unwrap(), value1);
        list.pop_back();
        assert_eq!(list.empty(), true);
    }

    #[test]
    fn size_linked_list() {
        let value1: f32 = 1.0;
        let value2: f32 = 2.0;
        let mut list = LinkedList::<f32>::new();
        assert_eq!(list.size(), 0);
        list.push_back(&value1);
        assert_eq!(list.size(), 1);
        list.push_back(&value2);
        assert_eq!(list.size(), 2);
        list.pop_back();
        assert_eq!(list.size(), 1);
        list.pop_back();
        assert_eq!(list.size(), 0);
    }
}
