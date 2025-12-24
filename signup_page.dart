import 'dart:io';
import 'package:airbnb_clone/common/common_functions.dart';
import 'package:airbnb_clone/pages/auth/login_page.dart';
import 'package:airbnb_clone/widgets/custom_text_field.dart';
import 'package:flutter/material.dart';
import '../../constants/app_constants.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:firebase_auth/firebase_auth.dart';


class SignupPage extends StatefulWidget {
  static const String routeName = '/signupPageRoute';

  const SignupPage({super.key});

  @override
  State<SignupPage> createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _firstNameController = TextEditingController();
  final _lastNameController = TextEditingController();
  final _cityController = TextEditingController();
  final _countryController = TextEditingController();
  final _bioController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  File? _imageFile;
  bool _isLoading = false;


  _chooseImage() async {
    // Pick image
    final XFile? pickedImage = await ImagePicker().pickImage(source: ImageSource.gallery);

    if (pickedImage != null) {
      // Convert XFile → File
      File originalFile = File(pickedImage.path);

      // Temporary directory for compressed file
      final tempDir = await getTemporaryDirectory();
      final targetPath = path.join(
        tempDir.path,
        "compressed_${DateTime.now().millisecondsSinceEpoch}.jpg",
      );

      // Compress image
      // We use compressWithFile and write manually to File to avoid XFile issue
      final compressedBytes = await FlutterImageCompress.compressWithFile(
        originalFile.path,
        quality: 25, // 25% quality
      );

      if (compressedBytes != null) {
        final compressedFile = File(targetPath)..writeAsBytesSync(compressedBytes);

        setState(() {
          _imageFile = compressedFile;
        });

        print("Original size: ${(originalFile.lengthSync() / 1024).toStringAsFixed(2)} KB");
        print("Compressed size: ${(_imageFile!.lengthSync() / 1024).toStringAsFixed(2)} KB");
      }
    }
  }

  _createAccount() async {
    if (!_formKey.currentState!.validate() || _imageFile == null) {
      CommonFunctions.showSnackBar(context, "Please fill all fields and choose a profile picture.");
      return;
    }

    setState(() {
      _isLoading = true;
    });

    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    try {
      UserCredential firebaseUser = await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: email,
          password: password
      );

      if (firebaseUser.user != null) {
        final userID = firebaseUser.user!.uid;
        AppConstants.currentUser.id = userID;
        AppConstants.currentUser.firstName = _firstNameController.text.trim();
        AppConstants.currentUser.lastName = _lastNameController.text.trim();
        AppConstants.currentUser.city = _cityController.text.trim();
        AppConstants.currentUser.country = _countryController.text.trim();
        AppConstants.currentUser.bio = _bioController.text.trim();
        AppConstants.currentUser.email = email;
        AppConstants.currentUser.password = password;

        await AppConstants.currentUser.addUserToFirestore();
        await AppConstants.currentUser.addImageToFirestore(_imageFile!);

        FirebaseAuth.instance.signOut();
        CommonFunctions.showSnackBar(context, "Your account created successfully. Please Login now.");

        _formKey.currentState!.reset();
        Navigator.pushReplacementNamed(context, LoginPage.routeName);
      }
    } on FirebaseAuthException catch(e) {
      String errorMessage;
      switch (e.code) {
        case "email-already-in-use":
          errorMessage = "This email is already registered.";
          break;
        case "invalid-email":
          errorMessage = "Please enter a valid email address.";
          break;
        case "weak-password":
          errorMessage = "Password is too weak. Please use a stronger one. Use at-least 6 characters";
          break;
        default:
          errorMessage = "Sign up failed. Please try again later.";
      }
      CommonFunctions.showSnackBar(context, errorMessage);
    } catch (e) {
      CommonFunctions.showSnackBar(context, e.toString());
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Create Account"),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(25),
        child: Column(
          children: [

            Image.asset(
                "assets/images/signup.png",
                width: MediaQuery.of(context).size.width * .8
            ),

            SizedBox(height: 15),

            Text(
              "Start Your Journey with\n${AppConstants.appName}",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 26,),
            ),

            Form(
              key: _formKey,
              child: Column(
                children: [

                  CustomTextField(controller: _emailController, label: "Email" , icon: Icons.email, isPassword: false, ),

                  CustomTextField(controller: _passwordController, label: "Password" , icon: Icons.lock, isPassword: true,),

                  CustomTextField(controller: _firstNameController, label: "First Name" , icon: Icons.person, isPassword: false,),

                  CustomTextField(controller: _lastNameController, label: "Last Name" , icon: Icons.person, isPassword: false,),

                  CustomTextField(controller: _cityController, label: "City" , icon: Icons.location_city, isPassword: false,),

                  CustomTextField(controller: _countryController, label: "Country" , icon: Icons.flag, isPassword: false,),

                  CustomTextField(
                    controller: _bioController,
                    label: "Tell Us a Little About You" ,
                    icon: Icons.info,
                    isPassword: false,
                    maxLines: 3,
                  ),

                ],
              ),
            ),

            SizedBox(height: 25),

            MaterialButton(
              onPressed: _chooseImage,
              child: (_imageFile == null)
                  ? const Icon(Icons.add_a_photo, size: 40, color: Colors.white)
                  : CircleAvatar(
                backgroundImage: FileImage(_imageFile!),
                radius: 60,
              ),
            ),

            SizedBox(height: 30),
            _isLoading
                ? const CircularProgressIndicator(color: Colors.white)
                : MaterialButton(
              onPressed: _createAccount,
              color: Colors.white,
              height: 55,
              minWidth: double.infinity,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: const Text(
                'Sign Up',
                style: TextStyle(
                  fontSize: 20,
                  color: Colors.black,
                ),
              ),
            ),

            SizedBox(height:20),

          ],
        ),
      ),
    );
  }
}
