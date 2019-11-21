import { Component, OnInit, AfterViewInit, ViewChild}  from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Renderer2, Inject, ElementRef } from '@angular/core';
import { DOCUMENT } from '@angular/common'
import { Tutorial } from '../tutorial' 
import { MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { DialogComponent} from '../dialog/dialog.component'

export interface DialogData {
  animal: string;
  name: string;
}

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit{
  email:any;
  name = "sam";
  animal = "dog"
  constructor(private http: HttpClient, public dialog: MatDialog) {
  }
  tutorialList:Tutorial[];
  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  ngOnInit() {}
  onClick(content:string){
    var httpPrefix = "https://www.googleapis.com/customsearch/v1?q=";
    var httpPostfix = "&cx=007435124061301021685%3Anx5ivx9bz4c&key=AIzaSyBqaEXf6vE07xB4PONkHzCSEb69XDCSud8";

    this.http.get(httpPrefix+content+httpPostfix, this.structureRequestOptions).subscribe(data => {
      console.log(data);

      // data.items[n].Link, Snippet, Title, 
    })

  }

  onClick2(content){
    const dialogRef = this.dialog.open(DialogComponent, {
      width: '250px',
      data: {name: this.name, animal: this.animal}
    });
  }
}

