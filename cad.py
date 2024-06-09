import streamlit as st

class CandidateElimination:
    def _init_(self, num_attributes):
        self.S = [set() for _ in range(num_attributes)]  # most specific hypothesis
        self.G = [set() for _ in range(num_attributes)]  # most general hypothesis
        self.num_attributes = num_attributes

    def fit(self, data):
        for instance in data:
            target = instance[-1]  # Assuming the last attribute is the target

            if target == "yes":  # Positive example
                self.specialize(instance[:-1])
            else:  # Negative example
                self.generalize(instance[:-1])

    def specialize(self, instance):
        for i in range(len(instance)):
            if instance[i] not in self.S[i]:
                self.S[i].add(instance[i])

            # Remove more general hypotheses inconsistent with the example
            remove_indices = []
            for j in range(len(self.G[i])):
                if list(self.G[i])[j] not in instance:
                    remove_indices.append(j)
            for index in remove_indices:
                if index < len(self.G[i]):  # Check if index is still within bounds
                    self.G[i].pop(index)

    def generalize(self, instance):
        for i in range(len(instance)):
            # Remove more specific hypotheses inconsistent with the example
            remove_indices = []
            for j in range(len(self.S[i])):
                if list(self.S[i])[j] != instance[i]:
                    remove_indices.append(j)
            for index in remove_indices:
                if index < len(self.S[i]):  # Check if index is still within bounds
                    self.S[i].pop(index)

            if instance[i] not in self.G[i]:
                self.G[i].add(instance[i])

    def get_hypotheses(self):
        return self.S, self.G

def main():
    st.title("Candidate Elimination Algorithm")
    st.subheader("Team: Cyber Centurions")

    # Example dataset
    data = [
        ['sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'yes'],
        ['sunny', 'warm', 'high', 'strong', 'warm', 'same', 'yes'],
        ['rainy', 'cold', 'high', 'strong', 'warm', 'change', 'no'],
        ['sunny', 'warm', 'high', 'strong', 'cool', 'change', 'yes']
    ]

    st.subheader("Example Dataset:")
    st.table(data)

    ce = CandidateElimination(num_attributes=len(data[0])-1)
    ce.fit(data)
    S, G = ce.get_hypotheses()

    st.subheader("Final Specific Hypothesis:")
    for i, hypothesis_set in enumerate(S):
        st.write(f"Attribute {i+1}: {hypothesis_set}")

    st.subheader("Final General Hypothesis:")
    for i, hypothesis_set in enumerate(G):
        st.write(f"Attribute {i+1}: {hypothesis_set}")

if _name_ == "_main_":
    main()
