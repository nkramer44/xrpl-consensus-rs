use xrpl_consensus_core::{Ledger, LedgerIndex};

use crate::span::SpanTip;

/// Ancestry trie of ledgers.
/// 
/// A compressed trie tree that maintains validation support of recent ledgers
/// based on their ancestry.
/// 
/// The compressed trie structure comes from recognizing that ledger history
/// can be viewed as a string over the alphabet of ledger ids. That is,
/// a given ledger with sequence number `seq` defines a length `seq` string,
/// with i-th entry equal to the id of the ancestor ledger with sequence
/// number i. "Sequence" strings with a common prefix share those ancestor
/// ledgers in common. Tracking this ancestry information and relations across
/// all validated ledgers is done conveniently in a compressed trie. A node in
/// the trie is an ancestor of all its children. If a parent node has sequence
/// number `seq`, each child node has a different ledger starting at `seq+1`.
/// The compression comes from the invariant that any non-root node with 0 tip
/// support has either no children or multiple children. In other words, a
/// non-root 0-tip-support node can be combined with its single child.
/// 
/// Each node has a tipSupport, which is the number of current validations for
/// that particular ledger. The node's branch support is the sum of the tip
/// support and the branch support of that node's children:
/// 
/// ```text
/// node.branchSupport = node.tipSupport;
/// for child in node.children
///    node.branchSupport += child.branchSupport;
/// ```
/// 
/// The generic [`Ledger`] `T` type represents a ledger which has a unique history.
///
/// The unique history invariant of ledgers requires any ledgers that agree
/// on the id of a given sequence number agree on ALL ancestors before that
/// ledger:
/// 
/// ```text
/// Ledger a,b;
/// /// For all Seq s:
/// if a[s] == b[s];
///     for p in 0..s {
///         assert(a[p] == b[p]);
/// }
/// ```
pub trait LedgerTrie<T: Ledger> {
    /// Insert and/or increment the support for the given ledger.
    ///
    /// # Params
    /// **ledger** - A `T` and its ancestry.
    ///
    /// **count** - The count of support for this ledger.
    fn insert(&mut self, ledger: &T, count: Option<u32>);

    /// Decrease support for a ledger, removing and compressing if possible.
    ///
    /// # Params
    /// **ledger** - The ledger history to remove.
    ///
    /// **count** - The amount of tip support to remove.
    ///
    /// # Returns
    /// Whether a matching node was decremented and possibly removed.
    fn remove(&mut self, ledger: &T, count: Option<u32>) -> bool;

    /// Return the preferred ledger ID
    ///
    /// The preferred ledger is used to determine the working ledger
    /// for consensus amongst competing alternatives.
    ///
    /// Recall that each validator is normally validating a chain of ledgers,
    /// e.g. A->B->C->D. However, if due to network connectivity or other
    /// issues, validators generate different chains, ie
    ///
    /// ```text
    ///        /->C
    ///    A->B
    ///        \->D->E
    /// ```
    ///
    /// we need a way for validators to converge on the chain with the most
    /// support. We call this the preferred ledger.  Intuitively, the idea is to
    /// be conservative and only switch to a different branch when you see
    /// enough peer validations to *know* another branch won't have preferred
    /// support.
    ///
    /// The preferred ledger is found by walking this tree of validated ledgers
    /// starting from the common ancestor ledger.
    ///
    /// At each sequence number, we have
    ///
    ///    - The prior sequence preferred ledger, e.g. B.
    ///    - The (tip) support of ledgers with this sequence number,e.g. the
    ///      number of validators whose last validation was for C or D.
    ///    - The (branch) total support of all descendants of the current
    ///      sequence number ledgers, e.g. the branch support of D is the
    ///      tip support of D plus the tip support of E; the branch support of
    ///      C is just the tip support of C.
    ///    - The number of validators that have yet to validate a ledger
    ///      with this sequence number (uncommitted support). Uncommitted
    ///      includes all validators whose last sequence number is smaller than
    ///      our last issued sequence number, since due to asynchrony, we may
    ///      not have heard from those nodes yet.
    ///
    /// The preferred ledger for this sequence number is then the ledger
    /// with relative majority of support, where uncommitted support
    /// can be given to ANY ledger at that sequence number
    /// (including one not yet known). If no such preferred ledger exists, then
    /// the prior sequence preferred ledger is the overall preferred ledger.
    ///
    /// In this example, for D to be preferred, the number of validators
    /// supporting it or a descendant must exceed the number of validators
    /// supporting C _plus_ the current uncommitted support. This is because if
    /// all uncommitted validators end up validating C, that new support must
    /// be less than that for D to be preferred.
    ///
    /// If a preferred ledger does exist, then we continue with the next
    /// sequence using that ledger as the root.
    ///
    /// # Params
    /// **largest_issued**: The sequence number of the largest validation issued by this node.
    ///
    /// # Returns
    /// The `SpanTip` of the preferred ledger or `None` if no preferred ledger exists.
    fn get_preferred(&self, largest_issued: LedgerIndex) -> Option<SpanTip<T>>;

    /// Return count of tip support for the specific ledger.
    fn tip_support(&self, ledger: &T) -> u32;

    /// Return count of tip support for the specific ledger.
    fn branch_support(&self, ledger: &T) -> u32;
}

