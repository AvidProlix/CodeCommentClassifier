// This is an example file to demonstrate the classifier

using system;

// <summary> this is a common example of a summary in autodocumenting code </summary>
void method(int n){
    return n;
}

// todo Add validation for the user input.
string userName = "";
if (!string.IsNullOrEmpty(userName))
{
    Console.WriteLine($"Hello, {userName}!");
}
else
{
    Console.WriteLine("Name cannot be empty.");
}

// error trying to divide by zero, catch the exception
int number1 = 5;
int number2 = 0;
try {
    Console.WriteLine(number1 / number2);
}
catch (DivideByZeroException) {
    Console.WriteLine("Division of {0} by zero.", number1);
}

// warning, if this API call result ever changes, this will break downstream
static HttpClient client = new HttpClient();
HttpResponseMessage response = await client.PostAsJsonAsync(
                "api/products", product);


