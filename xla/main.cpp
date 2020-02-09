#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include <iostream>

using namespace xla;

int main() {
  auto *client = ClientLibrary::LocalClientOrDie();

  XlaBuilder builder("exp");
  auto x = ConstantR1<float>(
      &builder, {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  Exp(x);

  std::vector<float> expected = {8.1662,     7.4274e-02, 13.4637,    1.8316e-02,
                                 8.1662,     9.9742,     6.7379e-03, 4.0657e-01,
                                 9.0718e-02, 4.9530};

  auto status = builder.Build();
  if (!status.ok()) {
    std::cout << "An error occurred\n";
    return 1;
  }

  auto computation = std::move(status.ValueOrDie());

  auto return_value = client->ExecuteAndTransfer(computation, {}).ValueOrDie();

  absl::Span<float> result = return_value.data<float>();

  for (int i = 0; i < result.size(); ++i) {
    std::cout << expected[i] << " == " << result[i] << '\n';
  }
}